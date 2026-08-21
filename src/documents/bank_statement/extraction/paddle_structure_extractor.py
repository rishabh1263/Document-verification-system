"""
PP-StructureV3 Table/Layout Extractor.

Phase 2 - Document Intelligence / Extraction (Tier 2 fallback).

Responsibilities:
- run PP-StructureV3 (document layout + table recognition) on
  rendered page images
- convert detected table cells and text blocks into the SAME
  generic OCRToken / OCRPage / OCRExtractionResult contract used
  by ocr_extractor.py, so this module is a drop-in alternative
  OCR source for the existing spatial pipeline
  (LayoutReconstructor -> TransactionAssembler -> TransactionParser)
- preserve row/column table geometry as token bounding boxes so
  downstream spatial reconciliation still works
- lazily initialize the PP-StructureV3 pipeline
- degrade gracefully (raise RuntimeError) when paddleocr /
  PP-StructureV3 is not installed, so the orchestrator can fall
  back to the next tier instead of crashing the request

Important:
This module does NOT:
- decide when it should run (that belongs to
  bank_statement_extractor.py, the tier orchestrator)
- parse transactions
- identify bank-specific fields
- contain bank-specific coordinates or templates

Designed for:
- PaddleOCR / PaddleX 3.x  (PPStructureV3 pipeline)
- PyMuPDF for PDF page rendering
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image

from src.documents.bank_statement.extraction.ocr_extractor import (
    OCRExtractionResult,
    OCRPage,
    OCRToken,
)


class PaddleStructureExtractor:
    """
    Tier-2 OCR-compatible extractor backed by PP-StructureV3.

    Used when the plain PP-OCR (ocr_extractor.py) result is low
    quality or fails to resolve a transaction table - typically
    dense, multi-column, or ruled-table statement layouts where
    reading order matters more than raw text detection.
    """

    PDF_RENDER_SCALE = 2.0
    ENGINE_NAME = "pp-structurev3"

    # Layout block labels that should be treated as free text
    # rather than as table content. Kept generic / bank-independent.
    TEXT_BLOCK_LABELS = {
        "text",
        "title",
        "paragraph_title",
        "abstract",
        "content",
        "header",
        "footer",
        "number",
    }

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
        """
        Run PP-StructureV3 on an image or a rendered PDF and return
        a result shaped exactly like OCRExtractionResult, so it can
        be substituted for a plain PaddleOCR result anywhere in the
        pipeline.
        """

        if not file_bytes:
            raise ValueError(
                "File bytes are required for structure extraction."
            )

        detected_type = (detected_type or "").strip().lower()

        images: list[Image.Image]

        if detected_type in {"jpeg", "jpg", "png"}:
            images = [self._load_image(file_bytes)]

        elif detected_type == "pdf":
            images = self._render_pdf_pages(file_bytes)

        else:
            raise ValueError(
                "PP-StructureV3 extraction does not support "
                f"file type: {detected_type or 'unknown'}"
            )

        pipeline = self._get_pipeline()

        extracted_pages: list[OCRPage] = []
        document_parts: list[str] = []
        page_confidences: list[float] = []
        tables_detected = 0

        try:
            for index, image in enumerate(images):
                page_number = index + 1

                try:
                    page, table_count = self._structure_page(
                        pipeline=pipeline,
                        image=image,
                        page_number=page_number,
                    )
                finally:
                    try:
                        image.close()
                    except Exception:
                        pass

                tables_detected += table_count
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
            "pp_structure_image"
            if detected_type in {"jpeg", "jpg", "png"}
            else "pp_structure_pdf"
        )

        result = OCRExtractionResult(
            filename=filename,
            extraction_method=method,
            page_count=len(extracted_pages),
            text=document_text,
            text_char_count=len(document_text),
            pages=tuple(extracted_pages),
            engine=self.ENGINE_NAME,
            average_confidence=average_confidence,
        )

        # tables_detected is informational only; stash it as an
        # attribute rather than widening the shared dataclass.
        object.__setattr__(result, "tables_detected", tables_detected)

        return result

    # ========================================================
    # PAGE PROCESSING
    # ========================================================

    def _structure_page(
        self,
        pipeline: Any,
        image: Image.Image,
        page_number: int,
    ) -> tuple[OCRPage, int]:

        import numpy as np

        rgb_image = image.convert("RGB") if image.mode != "RGB" else image
        image_width, image_height = rgb_image.size
        image_array = np.asarray(rgb_image)

        try:
            results = pipeline.predict(image_array)
        except Exception as exc:
            raise RuntimeError(
                f"PP-StructureV3 failed on page {page_number}: {exc}"
            ) from exc

        tokens: list[OCRToken] = []
        table_count = 0

        for result in self._to_list(results):
            data = self._result_to_dict(result)
            if not data:
                continue

            blocks = (
                data.get("parsing_res_list")
                or data.get("layout_parsing_result")
                or data.get("res")
                or []
            )

            for block in self._to_list(blocks):
                block_data = self._result_to_dict(block) or (
                    block if isinstance(block, dict) else {}
                )

                if not block_data:
                    continue

                label = str(
                    block_data.get("block_label")
                    or block_data.get("label")
                    or ""
                ).strip().lower()

                if "table" in label:
                    new_tokens = self._tokens_from_table_block(
                        block_data,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    if new_tokens:
                        table_count += 1
                    tokens.extend(new_tokens)
                    continue

                tokens.extend(
                    self._tokens_from_text_block(
                        block_data,
                        image_width=image_width,
                        image_height=image_height,
                    )
                )

        # Defensive fallback: some PP-StructureV3 builds return
        # OCR sub-results directly (overall_ocr_res) when a page
        # has no detectable layout blocks (e.g. plain text page).
        if not tokens:
            tokens = self._tokens_from_overall_ocr(
                results,
                image_width=image_width,
                image_height=image_height,
            )

        lines = [token.text for token in tokens if token.text]
        text = "\n".join(lines).strip()

        confidences = [
            token.confidence
            for token in tokens
            if token.confidence is not None
        ]
        confidence = (
            sum(confidences) / len(confidences) if confidences else None
        )

        page = OCRPage(
            page_number=page_number,
            text=text,
            char_count=len(text),
            line_count=len(lines),
            confidence=confidence,
            image_width=image_width,
            image_height=image_height,
            tokens=tuple(tokens),
        )

        return page, table_count

    # ========================================================
    # TABLE -> TOKENS
    # ========================================================

    def _tokens_from_table_block(
        self,
        block_data: dict,
        image_width: int,
        image_height: int,
    ) -> list[OCRToken]:
        """
        Convert one detected table region into row/column-ordered
        OCRTokens.

        Cell text is emitted as one token per cell, positioned at
        the cell's bounding box, so downstream row/column
        reconciliation (LayoutReconstructor) can regroup cells the
        same way it regroups plain OCR words.

        PP-StructureV3 typically exposes either:
          - `table_res_list` / `pred_html` (HTML table markup), or
          - `cell_box_list` + `table_ocr_pred` with per-cell boxes.
        Both shapes are handled defensively; if only HTML is
        available, cells are recovered from the HTML grid without
        geometry (falling back to a single synthetic column of
        row tokens ordered top-to-bottom).
        """

        tokens: list[OCRToken] = []

        block_bbox = self._extract_bbox(block_data)

        cell_boxes = (
            block_data.get("cell_box_list")
            or block_data.get("table_cell_bboxes")
            or []
        )

        cell_texts = (
            block_data.get("table_ocr_pred", {}).get("rec_texts")
            if isinstance(block_data.get("table_ocr_pred"), dict)
            else None
        ) or block_data.get("cell_texts") or []

        cell_boxes = self._to_list(cell_boxes)
        cell_texts = self._to_list(cell_texts)

        if cell_boxes and cell_texts and len(cell_boxes) == len(cell_texts):
            for box, text in zip(cell_boxes, cell_texts):
                text = str(text).strip()
                if not text:
                    continue

                polygon = self._box_to_polygon(box)
                if polygon is None:
                    continue

                tokens.append(
                    self._build_token(
                        text=text,
                        confidence=None,
                        polygon=polygon,
                        image_width=image_width,
                        image_height=image_height,
                    )
                )

            if tokens:
                return tokens

        # Fallback: parse the HTML table grid and lay rows out
        # as evenly spaced synthetic tokens inside the block's
        # bounding box, preserving row order (and, best-effort,
        # column order) even without native cell geometry.
        html = block_data.get("pred_html") or block_data.get("html")

        if html and block_bbox:
            tokens.extend(
                self._tokens_from_table_html(
                    html=html,
                    block_bbox=block_bbox,
                    image_width=image_width,
                    image_height=image_height,
                )
            )

        return tokens

    def _tokens_from_table_html(
        self,
        html: str,
        block_bbox: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> list[OCRToken]:

        import re

        x_min, y_min, x_max, y_max = block_bbox

        row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
        cell_pattern = re.compile(
            r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL
        )
        tag_pattern = re.compile(r"<[^>]+>")

        rows = row_pattern.findall(html or "")
        if not rows:
            return []

        row_height = max(1.0, (y_max - y_min) / max(1, len(rows)))
        tokens: list[OCRToken] = []

        for row_index, row_html in enumerate(rows):
            cells = cell_pattern.findall(row_html)
            if not cells:
                continue

            col_width = max(1.0, (x_max - x_min) / max(1, len(cells)))
            row_top = y_min + row_index * row_height
            row_bottom = row_top + row_height

            for col_index, cell_html in enumerate(cells):
                text = tag_pattern.sub(" ", cell_html)
                text = text.replace("&nbsp;", " ").replace("&amp;", "&")
                text = " ".join(text.split()).strip()

                if not text:
                    continue

                col_left = x_min + col_index * col_width
                col_right = col_left + col_width

                polygon = (
                    (col_left, row_top),
                    (col_right, row_top),
                    (col_right, row_bottom),
                    (col_left, row_bottom),
                )

                tokens.append(
                    self._build_token(
                        text=text,
                        confidence=None,
                        polygon=polygon,
                        image_width=image_width,
                        image_height=image_height,
                    )
                )

        return tokens

    # ========================================================
    # TEXT BLOCK -> TOKENS
    # ========================================================

    def _tokens_from_text_block(
        self,
        block_data: dict,
        image_width: int,
        image_height: int,
    ) -> list[OCRToken]:

        text = str(
            block_data.get("block_content")
            or block_data.get("text")
            or ""
        ).strip()

        if not text:
            return []

        polygon = self._box_to_polygon(self._extract_bbox(block_data))
        if polygon is None:
            return [OCRToken(text=text, confidence=None)]

        return [
            self._build_token(
                text=text,
                confidence=None,
                polygon=polygon,
                image_width=image_width,
                image_height=image_height,
            )
        ]

    def _tokens_from_overall_ocr(
        self,
        results: Any,
        image_width: int,
        image_height: int,
    ) -> list[OCRToken]:
        """
        Recover plain OCR tokens from PP-StructureV3's internal
        OCR sub-result when no layout blocks were returned. Keeps
        this tier at least as useful as plain PP-OCR on pages with
        no detectable table/layout structure.
        """

        tokens: list[OCRToken] = []

        for result in self._to_list(results):
            data = self._result_to_dict(result)
            overall = data.get("overall_ocr_res")
            overall = self._result_to_dict(overall) if overall else {}

            texts = self._to_list(overall.get("rec_texts"))
            scores = self._to_list(overall.get("rec_scores"))
            polys = self._to_list(
                overall.get("rec_polys") or overall.get("dt_polys")
            )

            for index, text in enumerate(texts):
                text = str(text).strip()
                if not text:
                    continue

                confidence = None
                if index < len(scores) and self._is_number(scores[index]):
                    confidence = float(scores[index])

                polygon = None
                if index < len(polys):
                    polygon = self._box_to_polygon(polys[index])

                if polygon is None:
                    tokens.append(
                        OCRToken(text=text, confidence=confidence)
                    )
                else:
                    tokens.append(
                        self._build_token(
                            text=text,
                            confidence=confidence,
                            polygon=polygon,
                            image_width=image_width,
                            image_height=image_height,
                        )
                    )

        return tokens

    # ========================================================
    # PDF RENDERING (mirrors ocr_extractor.py)
    # ========================================================

    def _render_pdf_pages(self, file_bytes: bytes) -> list[Image.Image]:

        if not file_bytes.startswith(b"%PDF-"):
            raise ValueError(
                "Structure extractor received a non-PDF file."
            )

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
            raise ValueError(f"Unable to open PDF for structure parsing: {exc}") from exc

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
            raise ValueError(f"Unable to open image for structure parsing: {exc}") from exc

    # ========================================================
    # LAZY PIPELINE INITIALIZATION
    # ========================================================

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from paddleocr import PPStructureV3
        except ImportError as exc:
            raise RuntimeError(
                "PP-StructureV3 is not installed. Install it with: "
                "python -m pip install paddleocr paddlepaddle"
            ) from exc

        try:
            self._pipeline = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to initialize PP-StructureV3: {exc}"
            ) from exc

        return self._pipeline

    # ========================================================
    # GEOMETRY / RESULT HELPERS
    # ========================================================

    @staticmethod
    def _extract_bbox(block_data: dict):
        box = (
            block_data.get("block_bbox")
            or block_data.get("bbox")
            or block_data.get("box")
        )
        if not box:
            return None

        try:
            values = [float(v) for v in box]
        except (TypeError, ValueError):
            return None

        if len(values) < 4:
            return None

        return tuple(values[:4])

    @staticmethod
    def _box_to_polygon(box: Any):
        if box is None:
            return None

        try:
            values = list(box)
        except TypeError:
            return None

        if not values:
            return None

        # Already a polygon: list of (x, y) pairs.
        first = values[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            try:
                return tuple((float(x), float(y)) for x, y in values)
            except (TypeError, ValueError):
                return None

        # Flat [x_min, y_min, x_max, y_max].
        try:
            flat = [float(v) for v in values]
        except (TypeError, ValueError):
            return None

        if len(flat) < 4:
            return None

        x_min, y_min, x_max, y_max = flat[:4]
        return (
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        )

    @classmethod
    def _build_token(
        cls,
        text: str,
        confidence: float | None,
        polygon,
        image_width: int,
        image_height: int,
    ) -> OCRToken:

        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        width = max(0.0, x_max - x_min)
        height = max(0.0, y_max - y_min)

        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0

        safe_width = float(image_width) if image_width > 0 else None
        safe_height = float(image_height) if image_height > 0 else None

        def norm_x(v: float):
            return None if safe_width is None else cls._clamp01(v / safe_width)

        def norm_y(v: float):
            return None if safe_height is None else cls._clamp01(v / safe_height)

        return OCRToken(
            text=text,
            confidence=confidence,
            x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
            x_center=x_center, y_center=y_center,
            width=width, height=height,
            x_min_norm=norm_x(x_min), y_min_norm=norm_y(y_min),
            x_max_norm=norm_x(x_max), y_max_norm=norm_y(y_max),
            x_center_norm=norm_x(x_center), y_center_norm=norm_y(y_center),
            polygon=polygon,
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

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
            import numpy as np
            if isinstance(value, np.ndarray):
                return value.tolist()
        except ImportError:
            pass
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


paddle_structure_extractor = PaddleStructureExtractor()