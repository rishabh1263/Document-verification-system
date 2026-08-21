"""
Generic PDF Content Stream Analyzer.

Analyzes local PDF page characteristics that may indicate
content insertion, overlays, unusual PDF construction, or
possible post-generation modification.

This complements PageConsistencyAnalyzer.

PageConsistencyAnalyzer:
    compares whole-page structure.

ContentStreamAnalyzer:
    inspects local PDF content characteristics.

Evidence categories:

WEAK:
- isolated font
- isolated font size
- unusual span count
- unusual stream length

MODERATE:
- unusual content-stream count
- unusual XObject usage
- multiple weak anomalies on the same page

STRONG:
- overlapping text objects
- duplicate text overlays

Important:
- No OCR.
- No bank-specific templates.
- No transaction extraction.
- No hardcoded page numbers.
- Findings are indicators, not forensic proof.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from statistics import median

import fitz


# ============================================================
# Page Evidence
# ============================================================


@dataclass(frozen=True)
class ContentStreamPageEvidence:

    page_number: int

    content_stream_count: int
    content_stream_length: int

    font_names: tuple[str, ...]
    font_sizes: tuple[float, ...]

    text_span_count: int

    overlapping_span_count: int
    duplicate_overlay_count: int

    xobject_count: int
    form_xobject_count: int

    weak_anomaly_count: int
    moderate_anomaly_count: int
    strong_anomaly_count: int

    anomaly_score: int


# ============================================================
# Analysis Result
# ============================================================


@dataclass(frozen=True)
class ContentStreamAnalysisResult:

    pages_analyzed: int

    median_content_stream_count: float
    median_content_stream_length: float
    median_text_span_count: float

    dominant_fonts: tuple[str, ...]
    dominant_font_sizes: tuple[float, ...]

    stream_count_outlier_pages: tuple[int, ...]
    stream_length_outlier_pages: tuple[int, ...]

    isolated_font_pages: tuple[int, ...]
    isolated_font_size_pages: tuple[int, ...]

    text_span_outlier_pages: tuple[int, ...]

    overlapping_text_pages: tuple[int, ...]
    duplicate_overlay_pages: tuple[int, ...]

    xobject_outlier_pages: tuple[int, ...]
    form_xobject_pages: tuple[int, ...]

    weak_anomaly_pages: tuple[int, ...]
    moderate_anomaly_pages: tuple[int, ...]
    strong_anomaly_pages: tuple[int, ...]

    suspicious_pages: tuple[int, ...]

    total_local_anomalies: int

    local_consistency_score: float

    pages: tuple[
        ContentStreamPageEvidence,
        ...
    ]

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Analyzer
# ============================================================


class ContentStreamAnalyzer:

    """
    Generic local PDF anomaly analyzer.

    Important design rule:

    A font difference or font-size difference alone is not
    sufficient evidence of tampering.

    Legitimate statements often contain:
    - cover pages
    - summary pages
    - footers
    - legal pages
    - logos
    - different typography

    Therefore weak anomalies are preserved as observations,
    while stronger evidence is evaluated separately.
    """

    # --------------------------------------------------------
    # Outlier thresholds
    # --------------------------------------------------------

    STREAM_LENGTH_THRESHOLD = 0.75

    TEXT_SPAN_THRESHOLD = 0.70

    FONT_PAGE_RATIO = 0.20

    FONT_SIZE_PAGE_RATIO = 0.20

    MIN_OVERLAP_AREA_RATIO = 0.55

    # --------------------------------------------------------
    # Page classification thresholds
    # --------------------------------------------------------

    MIN_WEAK_ANOMALIES_FOR_MODERATE = 3

    # ========================================================
    # Public API
    # ========================================================

    def analyze(
        self,
        file_bytes: bytes,
    ) -> ContentStreamAnalysisResult:

        if not file_bytes:

            raise ValueError(
                "PDF bytes are required."
            )

        try:

            document = fitz.open(
                stream=BytesIO(file_bytes),
                filetype="pdf",
            )

        except Exception as exc:

            raise ValueError(
                "Unable to open PDF for content-stream "
                "analysis."
            ) from exc

        try:

            if document.needs_pass:

                raise ValueError(
                    "Password-protected PDF cannot be "
                    "analyzed for content-stream anomalies."
                )

            if document.page_count <= 0:

                raise ValueError(
                    "PDF contains no pages."
                )

            # =================================================
            # Analyze every page
            # =================================================

            raw_pages = []

            for index in range(
                document.page_count
            ):

                raw_pages.append(
                    self._analyze_page(
                        document=document,
                        page=document[index],
                        page_number=index + 1,
                    )
                )

            # =================================================
            # Document baselines
            # =================================================

            median_stream_count = median(
                page["content_stream_count"]
                for page in raw_pages
            )

            median_stream_length = median(
                page["content_stream_length"]
                for page in raw_pages
            )

            median_span_count = median(
                page["text_span_count"]
                for page in raw_pages
            )

            median_xobjects = median(
                page["xobject_count"]
                for page in raw_pages
            )

            # =================================================
            # Dominant typography
            # =================================================

            dominant_fonts = (
                self._dominant_values(
                    raw_pages,
                    key="font_names",
                    page_ratio=(
                        self.FONT_PAGE_RATIO
                    ),
                )
            )

            dominant_font_sizes = (
                self._dominant_values(
                    raw_pages,
                    key="font_sizes",
                    page_ratio=(
                        self.FONT_SIZE_PAGE_RATIO
                    ),
                )
            )

            # =================================================
            # Evidence collections
            # =================================================

            stream_count_outliers = []
            stream_length_outliers = []

            isolated_font_pages = []
            isolated_font_size_pages = []

            text_span_outlier_pages = []

            overlapping_text_pages = []
            duplicate_overlay_pages = []

            xobject_outlier_pages = []
            form_xobject_pages = []

            weak_anomaly_pages = []
            moderate_anomaly_pages = []
            strong_anomaly_pages = []

            suspicious_pages = []

            final_pages = []

            # =================================================
            # Score each page
            # =================================================

            for page in raw_pages:

                page_number = (
                    page["page_number"]
                )

                weak_count = 0
                moderate_count = 0
                strong_count = 0

                anomaly_score = 0

                # =============================================
                # WEAK EVIDENCE
                # =============================================

                # ---------------------------------------------
                # Stream length
                # ---------------------------------------------

                if self._relative_outlier(
                    page[
                        "content_stream_length"
                    ],
                    median_stream_length,
                    self.STREAM_LENGTH_THRESHOLD,
                ):

                    stream_length_outliers.append(
                        page_number
                    )

                    weak_count += 1
                    anomaly_score += 1

                # ---------------------------------------------
                # Isolated fonts
                # ---------------------------------------------

                unexpected_fonts = (
                    set(
                        page["font_names"]
                    )
                    - set(
                        dominant_fonts
                    )
                )

                if unexpected_fonts:

                    isolated_font_pages.append(
                        page_number
                    )

                    weak_count += 1
                    anomaly_score += 1

                # ---------------------------------------------
                # Isolated font sizes
                # ---------------------------------------------

                unexpected_sizes = (
                    set(
                        page["font_sizes"]
                    )
                    - set(
                        dominant_font_sizes
                    )
                )

                if unexpected_sizes:

                    isolated_font_size_pages.append(
                        page_number
                    )

                    weak_count += 1
                    anomaly_score += 1

                # ---------------------------------------------
                # Text span count
                # ---------------------------------------------

                if self._relative_outlier(
                    page[
                        "text_span_count"
                    ],
                    median_span_count,
                    self.TEXT_SPAN_THRESHOLD,
                ):

                    text_span_outlier_pages.append(
                        page_number
                    )

                    weak_count += 1
                    anomaly_score += 1

                # =============================================
                # MODERATE EVIDENCE
                # =============================================

                # ---------------------------------------------
                # Content-stream count
                # ---------------------------------------------

                if self._count_outlier(
                    page[
                        "content_stream_count"
                    ],
                    median_stream_count,
                ):

                    stream_count_outliers.append(
                        page_number
                    )

                    moderate_count += 1
                    anomaly_score += 2

                # ---------------------------------------------
                # XObject usage
                # ---------------------------------------------

                if self._count_outlier(
                    page[
                        "xobject_count"
                    ],
                    median_xobjects,
                ):

                    xobject_outlier_pages.append(
                        page_number
                    )

                    moderate_count += 1
                    anomaly_score += 2

                # ---------------------------------------------
                # Form XObjects
                #
                # Presence is recorded but NOT automatically
                # treated as suspicious.
                # ---------------------------------------------

                if (
                    page[
                        "form_xobject_count"
                    ]
                    > 0
                ):

                    form_xobject_pages.append(
                        page_number
                    )

                # =============================================
                # STRONG EVIDENCE
                # =============================================

                # ---------------------------------------------
                # Overlapping text
                # ---------------------------------------------

                if (
                    page[
                        "overlapping_span_count"
                    ]
                    > 0
                ):

                    overlapping_text_pages.append(
                        page_number
                    )

                    strong_count += 1
                    anomaly_score += 4

                # ---------------------------------------------
                # Duplicate overlay
                # ---------------------------------------------

                if (
                    page[
                        "duplicate_overlay_count"
                    ]
                    > 0
                ):

                    duplicate_overlay_pages.append(
                        page_number
                    )

                    strong_count += 1
                    anomaly_score += 5

                # =============================================
                # PAGE EVIDENCE CLASSIFICATION
                # =============================================

                if weak_count > 0:

                    weak_anomaly_pages.append(
                        page_number
                    )

                # Several weak anomalies on one page can
                # become moderate evidence, but still not
                # strong evidence by themselves.

                weak_combination = (
                    weak_count
                    >= self
                    .MIN_WEAK_ANOMALIES_FOR_MODERATE
                )

                if (
                    moderate_count > 0
                    or weak_combination
                ):

                    moderate_anomaly_pages.append(
                        page_number
                    )

                if strong_count > 0:

                    strong_anomaly_pages.append(
                        page_number
                    )

                # Suspicious page requires either:
                #
                # 1. strong local evidence
                # OR
                # 2. multiple moderate signals
                #
                # A couple of typography differences are
                # deliberately insufficient.

                page_suspicious = bool(

                    strong_count > 0

                    or moderate_count >= 2

                    or (
                        moderate_count >= 1
                        and weak_count >= 2
                    )
                )

                if page_suspicious:

                    suspicious_pages.append(
                        page_number
                    )

                # =============================================
                # Store page result
                # =============================================

                final_pages.append(

                    ContentStreamPageEvidence(

                        page_number=(
                            page_number
                        ),

                        content_stream_count=(
                            page[
                                "content_stream_count"
                            ]
                        ),

                        content_stream_length=(
                            page[
                                "content_stream_length"
                            ]
                        ),

                        font_names=tuple(
                            page[
                                "font_names"
                            ]
                        ),

                        font_sizes=tuple(
                            page[
                                "font_sizes"
                            ]
                        ),

                        text_span_count=(
                            page[
                                "text_span_count"
                            ]
                        ),

                        overlapping_span_count=(
                            page[
                                "overlapping_span_count"
                            ]
                        ),

                        duplicate_overlay_count=(
                            page[
                                "duplicate_overlay_count"
                            ]
                        ),

                        xobject_count=(
                            page[
                                "xobject_count"
                            ]
                        ),

                        form_xobject_count=(
                            page[
                                "form_xobject_count"
                            ]
                        ),

                        weak_anomaly_count=(
                            weak_count
                        ),

                        moderate_anomaly_count=(
                            moderate_count
                        ),

                        strong_anomaly_count=(
                            strong_count
                        ),

                        anomaly_score=(
                            anomaly_score
                        ),
                    )
                )

            # =================================================
            # Document score
            # =================================================

            page_count = len(
                final_pages
            )

            suspicious_count = len(
                suspicious_pages
            )

            local_consistency_score = (
                (
                    page_count
                    - suspicious_count
                )
                / page_count
            ) * 100

            total_local_anomalies = sum(
                page.anomaly_score
                for page in final_pages
            )

            # =================================================
            # Result
            # =================================================

            return ContentStreamAnalysisResult(

                pages_analyzed=(
                    page_count
                ),

                median_content_stream_count=float(
                    median_stream_count
                ),

                median_content_stream_length=float(
                    median_stream_length
                ),

                median_text_span_count=float(
                    median_span_count
                ),

                dominant_fonts=tuple(
                    str(value)
                    for value
                    in dominant_fonts
                ),

                dominant_font_sizes=tuple(
                    float(value)
                    for value
                    in dominant_font_sizes
                ),

                stream_count_outlier_pages=tuple(
                    stream_count_outliers
                ),

                stream_length_outlier_pages=tuple(
                    stream_length_outliers
                ),

                isolated_font_pages=tuple(
                    isolated_font_pages
                ),

                isolated_font_size_pages=tuple(
                    isolated_font_size_pages
                ),

                text_span_outlier_pages=tuple(
                    text_span_outlier_pages
                ),

                overlapping_text_pages=tuple(
                    overlapping_text_pages
                ),

                duplicate_overlay_pages=tuple(
                    duplicate_overlay_pages
                ),

                xobject_outlier_pages=tuple(
                    xobject_outlier_pages
                ),

                form_xobject_pages=tuple(
                    form_xobject_pages
                ),

                weak_anomaly_pages=tuple(
                    weak_anomaly_pages
                ),

                moderate_anomaly_pages=tuple(
                    moderate_anomaly_pages
                ),

                strong_anomaly_pages=tuple(
                    strong_anomaly_pages
                ),

                suspicious_pages=tuple(
                    suspicious_pages
                ),

                total_local_anomalies=(
                    total_local_anomalies
                ),

                local_consistency_score=round(
                    local_consistency_score,
                    2,
                ),

                pages=tuple(
                    final_pages
                ),
            )

        finally:

            document.close()

    # ========================================================
    # Individual Page Analysis
    # ========================================================

    def _analyze_page(
        self,
        *,
        document: fitz.Document,
        page: fitz.Page,
        page_number: int,
    ) -> dict:

        # ====================================================
        # Content Streams
        # ====================================================

        try:

            content_xrefs = (
                page.get_contents()
            )

            if isinstance(
                content_xrefs,
                int,
            ):

                content_xrefs = [
                    content_xrefs
                ]

            content_xrefs = (
                content_xrefs or []
            )

        except Exception:

            content_xrefs = []

        content_stream_length = 0

        for xref in content_xrefs:

            try:

                stream = (
                    document.xref_stream(
                        xref
                    )
                )

                if stream:

                    content_stream_length += len(
                        stream
                    )

            except Exception:

                continue

        # ====================================================
        # Text Spans
        # ====================================================

        spans = []

        font_names = set()
        font_sizes = set()

        try:

            text_dict = (
                page.get_text(
                    "dict"
                )
            )

        except Exception:

            text_dict = {
                "blocks": []
            }

        for block in text_dict.get(
            "blocks",
            []
        ):

            if (
                block.get(
                    "type"
                )
                != 0
            ):

                continue

            for line in block.get(
                "lines",
                []
            ):

                for span in line.get(
                    "spans",
                    []
                ):

                    text = str(
                        span.get(
                            "text",
                            "",
                        )
                    ).strip()

                    if not text:

                        continue

                    bbox = span.get(
                        "bbox"
                    )

                    if not bbox:

                        continue

                    font = str(
                        span.get(
                            "font",
                            "",
                        )
                    ).strip()

                    size = round(
                        float(
                            span.get(
                                "size",
                                0.0,
                            )
                        ),
                        1,
                    )

                    if font:

                        font_names.add(
                            font
                        )

                    if size > 0:

                        font_sizes.add(
                            size
                        )

                    spans.append(
                        {
                            "text": text,

                            "bbox": tuple(
                                float(value)
                                for value
                                in bbox
                            ),

                            "font": font,

                            "size": size,
                        }
                    )

        # ====================================================
        # Overlap Analysis
        # ====================================================

        (
            overlap_count,
            duplicate_count,
        ) = self._detect_overlaps(
            spans
        )

        # ====================================================
        # XObjects
        # ====================================================

        xobject_count = 0
        form_xobject_count = 0

        try:

            xobjects = (
                page.get_xobjects()
            )

            xobject_count = len(
                xobjects
            )

            for item in xobjects:

                item_text = " ".join(
                    str(value)
                    for value in item
                ).lower()

                if "form" in item_text:

                    form_xobject_count += 1

        except Exception:

            pass

        return {

            "page_number": (
                page_number
            ),

            "content_stream_count": len(
                content_xrefs
            ),

            "content_stream_length": (
                content_stream_length
            ),

            "font_names": tuple(
                sorted(
                    font_names
                )
            ),

            "font_sizes": tuple(
                sorted(
                    font_sizes
                )
            ),

            "text_span_count": len(
                spans
            ),

            "overlapping_span_count": (
                overlap_count
            ),

            "duplicate_overlay_count": (
                duplicate_count
            ),

            "xobject_count": (
                xobject_count
            ),

            "form_xobject_count": (
                form_xobject_count
            ),
        }

    # ========================================================
    # Text Overlap Detection
    # ========================================================

    def _detect_overlaps(
        self,
        spans: list[dict],
    ) -> tuple[int, int]:

        overlap_count = 0
        duplicate_count = 0

        max_spans = min(
            len(spans),
            500,
        )

        for first_index in range(
            max_spans
        ):

            first = spans[
                first_index
            ]

            for second_index in range(
                first_index + 1,
                max_spans,
            ):

                second = spans[
                    second_index
                ]

                ratio = (
                    self._overlap_ratio(
                        first["bbox"],
                        second["bbox"],
                    )
                )

                if (
                    ratio
                    < self.MIN_OVERLAP_AREA_RATIO
                ):

                    continue

                overlap_count += 1

                if (
                    first["text"]
                    == second["text"]
                ):

                    duplicate_count += 1

        return (
            overlap_count,
            duplicate_count,
        )

    # ========================================================
    # Rectangle Overlap
    # ========================================================

    @staticmethod
    def _overlap_ratio(
        first: tuple[float, ...],
        second: tuple[float, ...],
    ) -> float:

        ax0, ay0, ax1, ay1 = first
        bx0, by0, bx1, by1 = second

        ix0 = max(
            ax0,
            bx0,
        )

        iy0 = max(
            ay0,
            by0,
        )

        ix1 = min(
            ax1,
            bx1,
        )

        iy1 = min(
            ay1,
            by1,
        )

        width = max(
            0.0,
            ix1 - ix0,
        )

        height = max(
            0.0,
            iy1 - iy0,
        )

        intersection = (
            width
            * height
        )

        if intersection <= 0:

            return 0.0

        first_area = max(
            (
                ax1
                - ax0
            )
            * (
                ay1
                - ay0
            ),
            0.0001,
        )

        second_area = max(
            (
                bx1
                - bx0
            )
            * (
                by1
                - by0
            ),
            0.0001,
        )

        smaller_area = min(
            first_area,
            second_area,
        )

        return (
            intersection
            / smaller_area
        )

    # ========================================================
    # Dominant Values
    # ========================================================

    @staticmethod
    def _dominant_values(
        pages: list[dict],
        *,
        key: str,
        page_ratio: float,
    ) -> tuple:

        counts = {}

        for page in pages:

            for value in set(
                page[key]
            ):

                counts[value] = (
                    counts.get(
                        value,
                        0,
                    )
                    + 1
                )

        minimum_pages = max(
            2,
            int(
                len(pages)
                * page_ratio
            ),
        )

        values = [

            value

            for value, count
            in counts.items()

            if count >= minimum_pages
        ]

        return tuple(
            sorted(
                values
            )
        )

    # ========================================================
    # Relative Outlier
    # ========================================================

    @staticmethod
    def _relative_outlier(
        value: float,
        baseline: float,
        threshold: float,
    ) -> bool:

        if baseline <= 0:

            return False

        difference = abs(
            value
            - baseline
        )

        return (
            difference
            / baseline
        ) > threshold

    # ========================================================
    # Count Outlier
    # ========================================================

    @staticmethod
    def _count_outlier(
        value: int,
        baseline: float,
    ) -> bool:

        if baseline <= 0:

            return value >= 2

        return (
            value
            > max(
                baseline * 2.5,
                baseline + 2,
            )
        )


# ============================================================
# Default Instance
# ============================================================


content_stream_analyzer = (
    ContentStreamAnalyzer()
)