"""
Generic OCR Extractor.

Phase 2 - Document Intelligence / Extraction.

Responsibilities:
- OCR JPEG/PNG bank-statement images
- OCR scanned/image-based PDF pages
- use PaddleOCR as the default OCR backend
- use PyMuPDF for PDF page rendering
- process PDF pages sequentially to control memory usage
- preserve page boundaries
- preserve OCR confidence information
- preserve OCR token geometry / bounding boxes
- return standardized bank-independent results
- lazily initialize PaddleOCR

Important:
This module does NOT:
- decide whether OCR is required
- parse transactions
- identify bank-specific fields
- detect tampering
- calculate fraud/risk scores
- contain bank-specific coordinates or templates

Routing decisions belong to extraction_router.py.

Designed for:
- PaddleOCR 3.x
- PyMuPDF
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


# ============================================================
# RESULT MODELS
# ============================================================


@dataclass(frozen=True)
class OCRToken:
    """
    One OCR-recognized text element with spatial information.

    Coordinates are expressed in rendered-image pixels.

    Normalized coordinates are also included so downstream layout
    reconstruction does not depend on:
    - PDF page size
    - image resolution
    - DPI
    - bank
    - statement template

    Example:
        text="Balance"
        x_center_norm=0.88

    means the token is located around 88% across the page width.
    """

    text: str
    confidence: float | None

    # Bounding rectangle in rendered image coordinates.
    x_min: float | None = None
    y_min: float | None = None
    x_max: float | None = None
    y_max: float | None = None

    # Center point.
    x_center: float | None = None
    y_center: float | None = None

    # Size.
    width: float | None = None
    height: float | None = None

    # Coordinates normalized to page dimensions [0, 1].
    x_min_norm: float | None = None
    y_min_norm: float | None = None
    x_max_norm: float | None = None
    y_max_norm: float | None = None
    x_center_norm: float | None = None
    y_center_norm: float | None = None

    # Original polygon returned by OCR where available.
    polygon: tuple[tuple[float, float], ...] = field(
        default_factory=tuple
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OCRPage:
    page_number: int
    text: str
    char_count: int
    line_count: int
    confidence: float | None

    # New layout-aware fields.
    image_width: int | None = None
    image_height: int | None = None
    tokens: tuple[OCRToken, ...] = field(default_factory=tuple)

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OCRExtractionResult:
    filename: str
    extraction_method: str
    page_count: int
    text: str
    text_char_count: int
    pages: tuple[OCRPage, ...]
    engine: str
    average_confidence: float | None

    @property
    def token_count(self) -> int:
        return sum(page.token_count for page in self.pages)

    @property
    def layout_available(self) -> bool:
        return any(
            token.x_center_norm is not None
            for page in self.pages
            for token in page.tokens
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# OCR EXTRACTOR
# ============================================================


class OCRExtractor:
    """
    Generic OCR extraction service.

    Current OCR backend:
        PaddleOCR 3.x

    PDF renderer:
        PyMuPDF

    Supported inputs:
        JPEG
        PNG
        scanned/image-based PDF

    PaddleOCR initialization is lazy.

    Digital PDFs routed through native extraction therefore do not
    load OCR models unnecessarily.

    IMPORTANT:
    This extractor preserves both:

        1. flattened text
        2. spatial OCR tokens

    Existing downstream code can continue using .text.

    Layout-aware downstream components can use .pages[].tokens.
    """

    # 2x zoom ~= 144 DPI for a standard 72-DPI PDF.
    #
    # We intentionally keep this unchanged for now so this upgrade
    # does not simultaneously alter OCR resolution and geometry.
    PDF_RENDER_SCALE = 2.0

    ENGINE_NAME = "paddleocr"

    SUPPORTED_IMAGE_TYPES = {
        "jpeg",
        "jpg",
        "png",
    }

    def __init__(self) -> None:
        self._ocr_engine = None

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
        OCR a supported image or scanned PDF.

        detected_type should normally come from Phase 1 /
        extraction_router.py.
        """

        if not file_bytes:
            raise ValueError(
                "File bytes are required for OCR extraction."
            )

        if not filename or not filename.strip():
            raise ValueError(
                "Filename is required."
            )

        detected_type = (
            detected_type or ""
        ).strip().lower()

        if detected_type in self.SUPPORTED_IMAGE_TYPES:
            return self._extract_image(
                file_bytes=file_bytes,
                filename=filename,
            )

        if detected_type == "pdf":
            return self._extract_pdf(
                file_bytes=file_bytes,
                filename=filename,
            )

        raise ValueError(
            "OCR extraction does not support file type: "
            f"{detected_type or 'unknown'}"
        )

    # ========================================================
    # IMAGE OCR
    # ========================================================

    def _extract_image(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> OCRExtractionResult:

        try:
            with Image.open(
                BytesIO(file_bytes)
            ) as source_image:

                source_image.load()

                image = source_image.convert(
                    "RGB"
                )

        except Exception as exc:
            raise ValueError(
                f"Unable to open image for OCR: {exc}"
            ) from exc

        try:
            page = self._ocr_image(
                image=image,
                page_number=1,
            )

        finally:
            try:
                image.close()
            except Exception:
                pass

        return OCRExtractionResult(
            filename=filename,
            extraction_method="ocr_image",
            page_count=1,
            text=page.text,
            text_char_count=page.char_count,
            pages=(page,),
            engine=self.ENGINE_NAME,
            average_confidence=page.confidence,
        )

    # ========================================================
    # SCANNED PDF OCR
    # ========================================================

    def _extract_pdf(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> OCRExtractionResult:
        """
        Render and OCR PDF pages sequentially.

        We deliberately avoid rendering all pages simultaneously.
        Large statements may contain dozens or hundreds of pages.
        """

        if not file_bytes.startswith(
            b"%PDF-"
        ):
            raise ValueError(
                "OCR PDF extractor received a non-PDF file."
            )

        try:
            import fitz

        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is not installed. "
                "Install it with: "
                "python -m pip install pymupdf"
            ) from exc

        try:
            document = fitz.open(
                stream=file_bytes,
                filetype="pdf",
            )

        except Exception as exc:
            raise ValueError(
                f"Unable to open PDF for OCR: {exc}"
            ) from exc

        try:
            page_count = document.page_count

            if page_count == 0:
                raise ValueError(
                    "PDF contains no readable pages."
                )

            extracted_pages: list[OCRPage] = []
            document_parts: list[str] = []
            page_confidences: list[float] = []

            for page_index in range(page_count):

                page_number = page_index + 1

                try:
                    image = self._render_pdf_page(
                        document=document,
                        page_index=page_index,
                    )

                except Exception as exc:
                    raise RuntimeError(
                        "Unable to render PDF page "
                        f"{page_number}: {exc}"
                    ) from exc

                try:
                    ocr_page = self._ocr_image(
                        image=image,
                        page_number=page_number,
                    )

                finally:
                    try:
                        image.close()
                    except Exception:
                        pass

                extracted_pages.append(
                    ocr_page
                )

                if ocr_page.text:
                    document_parts.append(
                        ocr_page.text
                    )

                if ocr_page.confidence is not None:
                    page_confidences.append(
                        ocr_page.confidence
                    )

            document_text = "\n\n".join(
                document_parts
            )

            average_confidence = (
                sum(page_confidences)
                / len(page_confidences)
                if page_confidences
                else None
            )

            return OCRExtractionResult(
                filename=filename,
                extraction_method="ocr_pdf",
                page_count=len(extracted_pages),
                text=document_text,
                text_char_count=len(document_text),
                pages=tuple(extracted_pages),
                engine=self.ENGINE_NAME,
                average_confidence=average_confidence,
            )

        finally:
            document.close()

    # ========================================================
    # PDF RENDERING
    # ========================================================

    def _render_pdf_page(
        self,
        document: Any,
        page_index: int,
    ) -> Image.Image:
        """
        Render one PDF page into RGB.

        Only one page exists in memory at a time.
        """

        try:
            import fitz

        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is not installed."
            ) from exc

        page = document.load_page(
            page_index
        )

        matrix = fitz.Matrix(
            self.PDF_RENDER_SCALE,
            self.PDF_RENDER_SCALE,
        )

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        if pixmap.n == 1:
            mode = "L"

        elif pixmap.n == 3:
            mode = "RGB"

        elif pixmap.n == 4:
            mode = "RGBA"

        else:
            raise RuntimeError(
                "Unsupported rendered PDF "
                f"channel count: {pixmap.n}"
            )

        image = Image.frombytes(
            mode,
            (
                pixmap.width,
                pixmap.height,
            ),
            pixmap.samples,
        )

        if image.mode != "RGB":
            image = image.convert(
                "RGB"
            )

        return image

    # ========================================================
    # OCR EXECUTION
    # ========================================================

    def _ocr_image(
        self,
        image: Image.Image,
        page_number: int,
    ) -> OCRPage:

        engine = self._get_engine()

        prepared_image = self._prepare_image(
            image
        )

        image_width = prepared_image.width
        image_height = prepared_image.height

        image_array = np.asarray(
            prepared_image
        )

        try:
            results = engine.predict(
                image_array
            )

        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR failed on page "
                f"{page_number}: {exc}"
            ) from exc

        tokens = self._parse_predict_results(
            results=results,
            image_width=image_width,
            image_height=image_height,
        )

        # Preserve OCR engine order for compatibility with the
        # previous implementation.
        lines = [
            token.text
            for token in tokens
            if token.text
        ]

        text = "\n".join(
            lines
        ).strip()

        confidences = [
            token.confidence
            for token in tokens
            if token.confidence is not None
        ]

        confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else None
        )

        return OCRPage(
            page_number=page_number,
            text=text,
            char_count=len(text),
            line_count=len(lines),
            confidence=confidence,
            image_width=image_width,
            image_height=image_height,
            tokens=tuple(tokens),
        )

    # ========================================================
    # LAZY PADDLE INITIALIZATION
    # ========================================================

    def _get_engine(self):
        """
        Initialize PaddleOCR only when OCR is required.
        """

        if self._ocr_engine is not None:
            return self._ocr_engine

        try:
            from paddleocr import PaddleOCR

        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed."
            ) from exc

        try:
            self._ocr_engine = PaddleOCR(
                lang="en",

                # Keep these disabled for speed.
                # Standard bank statements are normally upright.
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

        except Exception as exc:
            raise RuntimeError(
                "Unable to initialize PaddleOCR: "
                f"{exc}"
            ) from exc

        return self._ocr_engine

    # ========================================================
    # PADDLE RESULT ADAPTER
    # ========================================================

    @classmethod
    def _parse_predict_results(
        cls,
        results: Any,
        image_width: int,
        image_height: int,
    ) -> list[OCRToken]:
        """
        Convert PaddleOCR/PaddleX output into generic OCRToken
        objects.

        Paddle-specific structures stop here.

        Downstream components must work only with OCRToken and must
        never depend directly on PaddleOCR result internals.

        Expected PaddleOCR 3.x fields may include:

            rec_texts
            rec_scores
            rec_polys
            dt_polys

        Different PaddleOCR 3.x versions expose slightly different
        structures, so several compatible geometry fields are
        checked defensively.
        """

        tokens: list[OCRToken] = []

        if results is None:
            return tokens

        try:
            result_items = list(
                results
            )

        except TypeError:
            result_items = [
                results
            ]

        for result in result_items:

            data = cls._result_to_dict(
                result
            )

            if not data:
                continue

            candidate = data

            nested_result = data.get(
                "res"
            )

            if isinstance(
                nested_result,
                dict,
            ):
                candidate = nested_result

            # --------------------------------------------
            # Text
            # --------------------------------------------

            texts = candidate.get(
                "rec_texts"
            )

            if texts is None:
                texts = candidate.get(
                    "texts"
                )

            texts = cls._to_list(
                texts
            )

            # --------------------------------------------
            # Confidence
            # --------------------------------------------

            scores = candidate.get(
                "rec_scores"
            )

            if scores is None:
                scores = candidate.get(
                    "scores"
                )

            scores = cls._to_list(
                scores
            )

            # --------------------------------------------
            # Geometry
            # --------------------------------------------

            polygons = cls._extract_polygon_collection(
                candidate
            )

            # --------------------------------------------
            # Build generic tokens
            # --------------------------------------------

            for index, raw_text in enumerate(
                texts
            ):

                text = str(
                    raw_text
                ).strip()

                if not text:
                    continue

                confidence = cls._safe_score(
                    scores[index]
                    if index < len(scores)
                    else None
                )

                polygon = (
                    cls._normalize_polygon(
                        polygons[index]
                    )
                    if index < len(polygons)
                    else ()
                )

                token = cls._build_token(
                    text=text,
                    confidence=confidence,
                    polygon=polygon,
                    image_width=image_width,
                    image_height=image_height,
                )

                tokens.append(
                    token
                )

        return tokens

    # ========================================================
    # GEOMETRY EXTRACTION
    # ========================================================

    @classmethod
    def _extract_polygon_collection(
        cls,
        candidate: dict,
    ) -> list:
        """
        Obtain OCR geometry without exposing Paddle structures
        downstream.

        Priority:

            rec_polys
            rec_boxes
            dt_polys
            boxes

        rec_polys is preferred because it normally corresponds
        directly to rec_texts in PaddleOCR 3.x.
        """

        possible_keys = (
            "rec_polys",
            "rec_boxes",
            "dt_polys",
            "boxes",
        )

        for key in possible_keys:

            value = candidate.get(
                key
            )

            if value is None:
                continue

            values = cls._to_list(
                value
            )

            if values:
                return values

        return []

    @classmethod
    def _normalize_polygon(
        cls,
        raw_polygon: Any,
    ) -> tuple[tuple[float, float], ...]:
        """
        Normalize Paddle geometry into:

            ((x1, y1), (x2, y2), ...)

        Supports:
        - quadrilateral polygons
        - numpy arrays
        - nested lists
        - [x1, y1, x2, y2] rectangles
        """

        if raw_polygon is None:
            return ()

        if isinstance(
            raw_polygon,
            np.ndarray,
        ):
            raw_polygon = raw_polygon.tolist()

        if not isinstance(
            raw_polygon,
            (list, tuple),
        ):
            return ()

        # Rectangle format:
        # [x1, y1, x2, y2]
        if (
            len(raw_polygon) == 4
            and all(
                cls._is_number(value)
                for value in raw_polygon
            )
        ):

            x1, y1, x2, y2 = [
                float(value)
                for value in raw_polygon
            ]

            return (
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
            )

        points: list[tuple[float, float]] = []

        for point in raw_polygon:

            if isinstance(
                point,
                np.ndarray,
            ):
                point = point.tolist()

            if not isinstance(
                point,
                (list, tuple),
            ):
                continue

            if len(point) < 2:
                continue

            if not (
                cls._is_number(point[0])
                and cls._is_number(point[1])
            ):
                continue

            points.append(
                (
                    float(point[0]),
                    float(point[1]),
                )
            )

        return tuple(
            points
        )

    @classmethod
    def _build_token(
        cls,
        text: str,
        confidence: float | None,
        polygon: tuple[tuple[float, float], ...],
        image_width: int,
        image_height: int,
    ) -> OCRToken:
        """
        Create a bank-independent spatial OCR token.
        """

        if not polygon:

            return OCRToken(
                text=text,
                confidence=confidence,
            )

        xs = [
            point[0]
            for point in polygon
        ]

        ys = [
            point[1]
            for point in polygon
        ]

        x_min = min(xs)
        x_max = max(xs)

        y_min = min(ys)
        y_max = max(ys)

        width = max(
            0.0,
            x_max - x_min,
        )

        height = max(
            0.0,
            y_max - y_min,
        )

        x_center = (
            x_min + x_max
        ) / 2.0

        y_center = (
            y_min + y_max
        ) / 2.0

        safe_width = (
            float(image_width)
            if image_width > 0
            else None
        )

        safe_height = (
            float(image_height)
            if image_height > 0
            else None
        )

        def norm_x(
            value: float,
        ) -> float | None:

            if safe_width is None:
                return None

            return cls._clamp01(
                value / safe_width
            )

        def norm_y(
            value: float,
        ) -> float | None:

            if safe_height is None:
                return None

            return cls._clamp01(
                value / safe_height
            )

        return OCRToken(
            text=text,
            confidence=confidence,

            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,

            x_center=x_center,
            y_center=y_center,

            width=width,
            height=height,

            x_min_norm=norm_x(x_min),
            y_min_norm=norm_y(y_min),
            x_max_norm=norm_x(x_max),
            y_max_norm=norm_y(y_max),

            x_center_norm=norm_x(
                x_center
            ),
            y_center_norm=norm_y(
                y_center
            ),

            polygon=polygon,
        )

    # ========================================================
    # SCORE HELPERS
    # ========================================================

    @staticmethod
    def _safe_score(
        value: Any,
    ) -> float | None:

        if value is None:
            return None

        try:
            score = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

        if not (
            0.0 <= score <= 1.0
        ):
            return None

        return score

    # ========================================================
    # PADDLE RESULT CONVERSION
    # ========================================================

    @staticmethod
    def _result_to_dict(
        result: Any,
    ) -> dict:
        """
        Convert PaddleOCR/PaddleX result representations into a
        dictionary.

        Different PaddleOCR 3.x versions may expose result data
        differently.
        """

        if result is None:
            return {}

        if isinstance(
            result,
            dict,
        ):
            return result

        # PaddleX Result commonly exposes .json.
        try:
            json_value = result.json

            if callable(
                json_value
            ):
                json_value = json_value()

            if isinstance(
                json_value,
                dict,
            ):
                return json_value

        except Exception:
            pass

        # Other releases may expose .to_dict().
        try:
            to_dict = getattr(
                result,
                "to_dict",
                None,
            )

            if callable(
                to_dict
            ):

                value = to_dict()

                if isinstance(
                    value,
                    dict,
                ):
                    return value

        except Exception:
            pass

        # Defensive fallback.
        try:
            value = dict(
                result
            )

            if isinstance(
                value,
                dict,
            ):
                return value

        except Exception:
            pass

        return {}

    # ========================================================
    # GENERIC CONVERSION HELPERS
    # ========================================================

    @staticmethod
    def _to_list(
        value: Any,
    ) -> list:

        if value is None:
            return []

        if isinstance(
            value,
            list,
        ):
            return value

        if isinstance(
            value,
            tuple,
        ):
            return list(
                value
            )

        if isinstance(
            value,
            np.ndarray,
        ):
            return value.tolist()

        # Strings must remain one value, not become character lists.
        if isinstance(
            value,
            str,
        ):
            return [
                value
            ]

        try:
            return list(
                value
            )

        except TypeError:
            return [
                value
            ]

    @staticmethod
    def _is_number(
        value: Any,
    ) -> bool:

        try:
            float(
                value
            )
            return True

        except (
            TypeError,
            ValueError,
        ):
            return False

    @staticmethod
    def _clamp01(
        value: float,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )

    # ========================================================
    # IMAGE PREPARATION
    # ========================================================

    @staticmethod
    def _prepare_image(
        image: Image.Image,
    ) -> Image.Image:
        """
        Conservative image preprocessing.

        PaddleOCR already performs text detection and recognition.

        Aggressive thresholding is intentionally avoided because
        bank statements depend heavily on small visual details:

        - decimal points
        - commas
        - thin digits
        - reference numbers
        - CR / DR markers
        - table text
        """

        if image.mode != "RGB":
            image = image.convert(
                "RGB"
            )

        return image


# ============================================================
# SINGLETON
# ============================================================


ocr_extractor = OCRExtractor()