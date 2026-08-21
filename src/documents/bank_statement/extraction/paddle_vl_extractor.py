"""
PaddleOCR-VL Extractor.

Phase 2 - Document Intelligence / Extraction (Tier 3 / last-resort
fallback).

Responsibilities:
- run PaddleOCR-VL (VLM-based document parsing) on rendered page
  images, for statements that both plain PP-OCR and PP-StructureV3
  failed to resolve into a usable transaction table (poor scans,
  unusual/handwritten layouts, heavily rotated or watermarked pages)
- return the page's markdown (tables preserved as markdown tables)
  as bank-independent flattened text, wrapped in the same
  OCRExtractionResult contract used elsewhere in the pipeline
- lazily initialize the PaddleOCR-VL pipeline
- degrade gracefully (raise RuntimeError) when PaddleOCR-VL is not
  installed / not available in this environment, so the
  orchestrator can report the earlier tier's result instead of
  crashing the request

Important:
This module does NOT:
- decide when it should run (that belongs to
  bank_statement_extractor.py, the tier orchestrator)
- parse transactions
- identify bank-specific fields
- return spatial token geometry - PaddleOCR-VL is a
  vision-language document parser, not a detector/recognizer, so
  its output is consumed through the TEXT pipeline
  (TextNormalizer -> StructureParser -> TransactionParser.parse),
  not the spatial pipeline used for plain OCR.

Designed for:
- PaddleOCR-VL, shipped as a PaddleX pipeline in PaddleOCR /
  PaddleX 3.x ("PaddleOCR-VL")
- PyMuPDF for PDF page rendering
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image

from src.documents.bank_statement.extraction.ocr_extractor import (
    OCRExtractionResult,
    OCRPage,
)


class PaddleVLExtractor:
    """
    Tier-3 OCR-compatible extractor backed by PaddleOCR-VL.

    Used only after both PP-OCR and PP-StructureV3 fail to produce
    a confident transaction table. PaddleOCR-VL is slower and more
    resource-intensive, so it is intentionally kept as a last
    resort rather than a default path.
    """

    PDF_RENDER_SCALE = 2.0
    ENGINE_NAME = "paddleocr-vl"

    def __init__(self) -> None:
        self._pipeline = None

    # ========================================================
    # PUBLIC API
    # ========================================================

    def extract(
        self,
        file_bytes: bytes,
        filename: str,
        detected_type: str,
    ) -> OCRExtractionResult:

        if not file_bytes:
            raise ValueError(
                "File bytes are required for PaddleOCR-VL extraction."
            )

        detected_type = (detected_type or "").strip().lower()

        if detected_type in {"jpeg", "jpg", "png"}:
            images = [self._load_image(file_bytes)]
        elif detected_type == "pdf":
            images = self._render_pdf_pages(file_bytes)
        else:
            raise ValueError(
                "PaddleOCR-VL extraction does not support file "
                f"type: {detected_type or 'unknown'}"
            )

        pipeline = self._get_pipeline()

        extracted_pages: list[OCRPage] = []
        document_parts: list[str] = []
        page_confidences: list[float] = []

        try:
            for index, image in enumerate(images):
                page_number = index + 1
                try:
                    page = self._parse_page(
                        pipeline=pipeline,
                        image=image,
                        page_number=page_number,
                    )
                finally:
                    try:
                        image.close()
                    except Exception:
                        pass

                extracted_pages.append(page)

                if page.text:
                    document_parts.append(page.text)

                if page.confidence is not None:
                    page_confidences.append(page.confidence)
        finally:
            images = []

        document_text = "\n\n".join(document_parts)

        average_confidence = (
            sum(page_confidences) / len(page_confidences)
            if page_confidences
            else None
        )

        method = (
            "paddleocr_vl_image"
            if detected_type in {"jpeg", "jpg", "png"}
            else "paddleocr_vl_pdf"
        )

        return OCRExtractionResult(
            filename=filename,
            extraction_method=method,
            page_count=len(extracted_pages),
            text=document_text,
            text_char_count=len(document_text),
            pages=tuple(extracted_pages),
            engine=self.ENGINE_NAME,
            average_confidence=average_confidence,
        )

    # ========================================================
    # PAGE PROCESSING
    # ========================================================

    def _parse_page(
        self,
        pipeline: Any,
        image: Image.Image,
        page_number: int,
    ) -> OCRPage:

        import numpy as np

        rgb_image = image.convert("RGB") if image.mode != "RGB" else image
        image_width, image_height = rgb_image.size
        image_array = np.asarray(rgb_image)

        try:
            results = pipeline.predict(image_array)
        except Exception as exc:
            raise RuntimeError(
                f"PaddleOCR-VL failed on page {page_number}: {exc}"
            ) from exc

        text, confidence = self._extract_markdown(results)

        lines = [line for line in text.splitlines() if line.strip()]

        return OCRPage(
            page_number=page_number,
            text=text.strip(),
            char_count=len(text.strip()),
            line_count=len(lines),
            confidence=confidence,
            image_width=image_width,
            image_height=image_height,
            tokens=(),
        )

    def _extract_markdown(self, results: Any) -> tuple[str, float | None]:
        """
        Recover flattened markdown text (with markdown tables
        preserved) from a PaddleOCR-VL prediction.

        Different PaddleX/PaddleOCR-VL releases expose the parsed
        document slightly differently, so several compatible
        fields are checked defensively.
        """

        parts: list[str] = []
        scores: list[float] = []

        for result in self._to_list(results):
            data = self._result_to_dict(result)
            if not data:
                continue

            markdown = data.get("markdown")

            if isinstance(markdown, dict):
                text = markdown.get("markdown_texts") or markdown.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(markdown, str) and markdown.strip():
                parts.append(markdown)

            if not markdown:
                # Fallback: plain parsed text / OCR text fields.
                fallback_text = (
                    data.get("parsing_res_text")
                    or data.get("text")
                    or data.get("content")
                )
                if fallback_text:
                    parts.append(str(fallback_text))

            score = data.get("score") or data.get("confidence")
            if self._is_number(score):
                scores.append(float(score))

        text = "\n\n".join(part for part in parts if part).strip()
        confidence = sum(scores) / len(scores) if scores else None

        return text, confidence

    # ========================================================
    # PDF RENDERING (mirrors ocr_extractor.py)
    # ========================================================

    def _render_pdf_pages(self, file_bytes: bytes) -> list[Image.Image]:

        if not file_bytes.startswith(b"%PDF-"):
            raise ValueError("PaddleOCR-VL extractor received a non-PDF file.")

        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is not installed. "
                "Install it with: python -m pip install pymupdf"
            ) from exc

        try:
            document = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"Unable to open PDF for PaddleOCR-VL: {exc}") from exc

        images: list[Image.Image] = []

        try:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                matrix = fitz.Matrix(
                    self.PDF_RENDER_SCALE, self.PDF_RENDER_SCALE
                )
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)

                mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(pixmap.n)
                if mode is None:
                    raise RuntimeError(
                        f"Unsupported rendered PDF channel count: {pixmap.n}"
                    )

                image = Image.frombytes(
                    mode, (pixmap.width, pixmap.height), pixmap.samples
                )
                if image.mode != "RGB":
                    image = image.convert("RGB")

                images.append(image)
        finally:
            document.close()

        return images

    @staticmethod
    def _load_image(file_bytes: bytes) -> Image.Image:
        try:
            with Image.open(BytesIO(file_bytes)) as source_image:
                source_image.load()
                return source_image.convert("RGB")
        except Exception as exc:
            raise ValueError(f"Unable to open image for PaddleOCR-VL: {exc}") from exc

    # ========================================================
    # LAZY PIPELINE INITIALIZATION
    # ========================================================

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        # PaddleOCR-VL is shipped as a PaddleX pipeline. Two
        # compatible entry points are attempted defensively, since
        # this is a newer/actively-evolving pipeline name.
        try:
            from paddleocr import PaddleOCRVL

            try:
                self._pipeline = PaddleOCRVL()
                return self._pipeline
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to initialize PaddleOCR-VL: {exc}"
                ) from exc

        except ImportError:
            pass

        try:
            from paddlex import create_pipeline
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR-VL is not installed. Install it with: "
                "python -m pip install paddleocr paddlex paddlepaddle "
                "(PaddleOCR-VL requires PaddleOCR/PaddleX >= 3.x)"
            ) from exc

        try:
            self._pipeline = create_pipeline(pipeline="PaddleOCR-VL")
        except Exception as exc:
            raise RuntimeError(
                f"Unable to initialize PaddleOCR-VL pipeline: {exc}"
            ) from exc

        return self._pipeline

    # ========================================================
    # RESULT HELPERS
    # ========================================================

    @staticmethod
    def _is_number(value: Any) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _to_list(value: Any) -> list:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, str):
            return [value]
        try:
            return list(value)
        except TypeError:
            return [value]

    @staticmethod
    def _result_to_dict(result: Any) -> dict:
        if result is None:
            return {}
        if isinstance(result, dict):
            return result

        try:
            json_value = result.json
            if callable(json_value):
                json_value = json_value()
            if isinstance(json_value, dict):
                return json_value.get("res", json_value)
        except Exception:
            pass

        try:
            to_dict = getattr(result, "to_dict", None)
            if callable(to_dict):
                value = to_dict()
                if isinstance(value, dict):
                    return value
        except Exception:
            pass

        try:
            value = dict(result)
            if isinstance(value, dict):
                return value
        except Exception:
            pass

        return {}


paddle_vl_extractor = PaddleVLExtractor()