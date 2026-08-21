"""
Generic PDF Object Profile Analyzer.

Purpose
-------
Analyze the internal object/resource construction of a PDF and identify
unusual page-level object patterns that may support tamper detection.

This module is intentionally:
- bank-independent
- OCR-free
- transaction-independent
- modular
- evidence-producing, not verdict-producing

It complements:
- PDFIntegrityAnalyzer
- PageConsistencyAnalyzer
- ContentStreamAnalyzer
- PDFRevisionAnalyzer

Important
---------
An unusual PDF object structure is NOT proof of tampering.

PDF generators, printers, scanners, converters, signing software and
document-management systems can legitimately produce unusual object
structures.

The output should therefore be correlated with other independent
evidence by TamperDetector.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from io import BytesIO
from statistics import median

import fitz


# ============================================================
# Page Profile
# ============================================================


@dataclass(frozen=True)
class PDFPageObjectProfile:

    page_number: int
    page_xref: int

    content_stream_count: int

    image_count: int
    font_count: int
    xobject_count: int

    annotation_count: int
    link_count: int
    drawing_count: int

    resource_signature: str

    object_complexity_score: int


# ============================================================
# Result
# ============================================================


@dataclass(frozen=True)
class PDFObjectProfileResult:

    pages_analyzed: int
    xref_length: int

    median_content_stream_count: float
    median_image_count: float
    median_font_count: float
    median_xobject_count: float
    median_drawing_count: float

    dominant_resource_signatures: tuple[str, ...]

    content_stream_outlier_pages: tuple[int, ...]
    image_object_outlier_pages: tuple[int, ...]
    font_object_outlier_pages: tuple[int, ...]
    xobject_outlier_pages: tuple[int, ...]
    drawing_object_outlier_pages: tuple[int, ...]

    annotation_pages: tuple[int, ...]
    link_pages: tuple[int, ...]

    rare_resource_signature_pages: tuple[int, ...]

    multi_signal_object_outlier_pages: tuple[int, ...]

    suspicious_object_profile: bool

    object_consistency_score: float

    weak_signals: tuple[str, ...]
    moderate_signals: tuple[str, ...]
    strong_signals: tuple[str, ...]
    warnings: tuple[str, ...]

    pages: tuple[PDFPageObjectProfile, ...]

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Analyzer
# ============================================================


class PDFObjectProfileAnalyzer:

    """
    Analyze PDF page-resource/object profiles.

    Design principle:

    A single unusual font/image/resource count is weak evidence.

    Multiple independent object anomalies on the same page are
    more useful because they indicate that the page was constructed
    differently from its peers.

    No bank-specific rules or page numbers are used.
    """

    RELATIVE_THRESHOLD = 0.80

    DOMINANT_SIGNATURE_RATIO = 0.10

    MIN_MULTI_SIGNAL_COUNT = 2

    # ========================================================
    # Public API
    # ========================================================

    def analyze(
        self,
        file_bytes: bytes,
    ) -> PDFObjectProfileResult:

        if not file_bytes:

            raise ValueError(
                "PDF bytes are required."
            )

        if not file_bytes.startswith(
            b"%PDF-"
        ):

            raise ValueError(
                "Input does not have a valid PDF signature."
            )

        try:

            document = fitz.open(
                stream=BytesIO(file_bytes),
                filetype="pdf",
            )

        except Exception as exc:

            raise ValueError(
                "Unable to open PDF for object-profile analysis."
            ) from exc

        try:

            if document.needs_pass:

                raise ValueError(
                    "Password-protected PDF cannot be fully "
                    "analyzed for object structure."
                )

            if document.page_count <= 0:

                raise ValueError(
                    "PDF contains no pages."
                )

            # =================================================
            # Build raw page profiles
            # =================================================

            raw_pages: list[dict] = []

            for index in range(
                document.page_count
            ):

                page = document[index]

                raw_pages.append(
                    self._analyze_page(
                        document=document,
                        page=page,
                        page_number=index + 1,
                    )
                )

            # =================================================
            # Baselines
            # =================================================

            median_streams = median(
                page["content_stream_count"]
                for page in raw_pages
            )

            median_images = median(
                page["image_count"]
                for page in raw_pages
            )

            median_fonts = median(
                page["font_count"]
                for page in raw_pages
            )

            median_xobjects = median(
                page["xobject_count"]
                for page in raw_pages
            )

            median_drawings = median(
                page["drawing_count"]
                for page in raw_pages
            )

            # =================================================
            # Resource signatures
            # =================================================

            signature_counts = Counter(
                page["resource_signature"]
                for page in raw_pages
            )

            minimum_dominant_pages = max(
                2,
                int(
                    len(raw_pages)
                    * self.DOMINANT_SIGNATURE_RATIO
                ),
            )

            dominant_signatures = tuple(
                sorted(
                    signature
                    for signature, count
                    in signature_counts.items()
                    if count >= minimum_dominant_pages
                )
            )

            # =================================================
            # Evidence containers
            # =================================================

            stream_outliers: list[int] = []
            image_outliers: list[int] = []
            font_outliers: list[int] = []
            xobject_outliers: list[int] = []
            drawing_outliers: list[int] = []

            annotation_pages: list[int] = []
            link_pages: list[int] = []

            rare_signature_pages: list[int] = []

            multi_signal_pages: list[int] = []

            final_pages: list[
                PDFPageObjectProfile
            ] = []

            weak_signals: list[str] = []
            moderate_signals: list[str] = []
            strong_signals: list[str] = []
            warnings: list[str] = []

            # =================================================
            # Analyze deviations
            # =================================================

            for page in raw_pages:

                page_number = (
                    page["page_number"]
                )

                anomaly_count = 0

                # ------------------------------------------------
                # Content streams
                # ------------------------------------------------

                if self._count_outlier(
                    page["content_stream_count"],
                    median_streams,
                ):

                    stream_outliers.append(
                        page_number
                    )

                    anomaly_count += 1

                # ------------------------------------------------
                # Images
                # ------------------------------------------------

                if self._count_outlier(
                    page["image_count"],
                    median_images,
                ):

                    image_outliers.append(
                        page_number
                    )

                    anomaly_count += 1

                # ------------------------------------------------
                # Fonts
                # ------------------------------------------------

                if self._relative_outlier(
                    page["font_count"],
                    median_fonts,
                ):

                    font_outliers.append(
                        page_number
                    )

                    anomaly_count += 1

                # ------------------------------------------------
                # XObjects
                # ------------------------------------------------

                if self._count_outlier(
                    page["xobject_count"],
                    median_xobjects,
                ):

                    xobject_outliers.append(
                        page_number
                    )

                    anomaly_count += 1

                # ------------------------------------------------
                # Drawings
                # ------------------------------------------------

                if self._relative_outlier(
                    page["drawing_count"],
                    median_drawings,
                ):

                    drawing_outliers.append(
                        page_number
                    )

                    anomaly_count += 1

                # ------------------------------------------------
                # Annotations
                # ------------------------------------------------

                if page["annotation_count"] > 0:

                    annotation_pages.append(
                        page_number
                    )

                # ------------------------------------------------
                # Links
                # ------------------------------------------------

                if page["link_count"] > 0:

                    link_pages.append(
                        page_number
                    )

                # ------------------------------------------------
                # Resource signature
                # ------------------------------------------------

                if (
                    dominant_signatures
                    and page["resource_signature"]
                    not in dominant_signatures
                ):

                    rare_signature_pages.append(
                        page_number
                    )

                    anomaly_count += 1

                # ------------------------------------------------
                # Multi-signal page
                # ------------------------------------------------

                if (
                    anomaly_count
                    >= self.MIN_MULTI_SIGNAL_COUNT
                ):

                    multi_signal_pages.append(
                        page_number
                    )

                final_pages.append(

                    PDFPageObjectProfile(

                        page_number=(
                            page_number
                        ),

                        page_xref=(
                            page["page_xref"]
                        ),

                        content_stream_count=(
                            page[
                                "content_stream_count"
                            ]
                        ),

                        image_count=(
                            page["image_count"]
                        ),

                        font_count=(
                            page["font_count"]
                        ),

                        xobject_count=(
                            page["xobject_count"]
                        ),

                        annotation_count=(
                            page["annotation_count"]
                        ),

                        link_count=(
                            page["link_count"]
                        ),

                        drawing_count=(
                            page["drawing_count"]
                        ),

                        resource_signature=(
                            page[
                                "resource_signature"
                            ]
                        ),

                        object_complexity_score=(
                            anomaly_count
                        ),
                    )
                )

            # =================================================
            # Evidence classification
            # =================================================

            if stream_outliers:

                weak_signals.append(
                    "Content-stream count variation detected "
                    "on page(s): "
                    + self._page_list(
                        stream_outliers
                    )
                    + "."
                )

            if image_outliers:

                weak_signals.append(
                    "Image-object variation detected on "
                    "page(s): "
                    + self._page_list(
                        image_outliers
                    )
                    + "."
                )

            if font_outliers:

                weak_signals.append(
                    "Font-resource count variation detected "
                    "on page(s): "
                    + self._page_list(
                        font_outliers
                    )
                    + "."
                )

            if xobject_outliers:

                moderate_signals.append(
                    "XObject-resource variation detected on "
                    "page(s): "
                    + self._page_list(
                        xobject_outliers
                    )
                    + "."
                )

            if drawing_outliers:

                weak_signals.append(
                    "Drawing-object variation detected on "
                    "page(s): "
                    + self._page_list(
                        drawing_outliers
                    )
                    + "."
                )

            if rare_signature_pages:

                weak_signals.append(
                    "Rare page-resource signatures detected "
                    "on page(s): "
                    + self._page_list(
                        rare_signature_pages
                    )
                    + "."
                )

            if annotation_pages:

                weak_signals.append(
                    "PDF annotations are present on page(s): "
                    + self._page_list(
                        annotation_pages
                    )
                    + "."
                )

            # Links are recorded but not automatically suspicious.
            #
            # Bank statements may legitimately contain links.

            if multi_signal_pages:

                moderate_signals.append(
                    "Multiple independent object-profile "
                    "variations occur on page(s): "
                    + self._page_list(
                        multi_signal_pages
                    )
                    + "."
                )

            # =================================================
            # Strong evidence
            # =================================================

            # Do not manufacture strong evidence from one odd
            # page. Require multiple pages with correlated object
            # profile deviations.

            if len(
                multi_signal_pages
            ) >= 3:

                strong_signals.append(
                    "Multiple pages contain correlated PDF "
                    "object-profile anomalies."
                )

            # =================================================
            # Object consistency score
            # =================================================

            suspicious_page_set = set(
                multi_signal_pages
            )

            page_count = len(
                raw_pages
            )

            object_consistency_score = (
                (
                    page_count
                    - len(
                        suspicious_page_set
                    )
                )
                / page_count
            ) * 100

            # =================================================
            # Suspicious profile decision
            # =================================================

            suspicious_object_profile = bool(
                strong_signals
                or (
                    len(
                        multi_signal_pages
                    )
                    >= 2
                    and len(
                        moderate_signals
                    )
                    >= 2
                )
            )

            # =================================================
            # Warnings
            # =================================================

            if (
                rare_signature_pages
                or multi_signal_pages
            ):

                warnings.append(
                    "Different PDF object/resource structures "
                    "can be produced legitimately by cover "
                    "pages, summaries, logos, signatures, "
                    "printing software, scanners, converters, "
                    "or document-management systems."
                )

            # =================================================
            # Result
            # =================================================

            return PDFObjectProfileResult(

                pages_analyzed=(
                    page_count
                ),

                xref_length=int(
                    document.xref_length()
                ),

                median_content_stream_count=float(
                    median_streams
                ),

                median_image_count=float(
                    median_images
                ),

                median_font_count=float(
                    median_fonts
                ),

                median_xobject_count=float(
                    median_xobjects
                ),

                median_drawing_count=float(
                    median_drawings
                ),

                dominant_resource_signatures=(
                    dominant_signatures
                ),

                content_stream_outlier_pages=tuple(
                    stream_outliers
                ),

                image_object_outlier_pages=tuple(
                    image_outliers
                ),

                font_object_outlier_pages=tuple(
                    font_outliers
                ),

                xobject_outlier_pages=tuple(
                    xobject_outliers
                ),

                drawing_object_outlier_pages=tuple(
                    drawing_outliers
                ),

                annotation_pages=tuple(
                    annotation_pages
                ),

                link_pages=tuple(
                    link_pages
                ),

                rare_resource_signature_pages=tuple(
                    rare_signature_pages
                ),

                multi_signal_object_outlier_pages=tuple(
                    multi_signal_pages
                ),

                suspicious_object_profile=(
                    suspicious_object_profile
                ),

                object_consistency_score=round(
                    object_consistency_score,
                    2,
                ),

                weak_signals=tuple(
                    weak_signals
                ),

                moderate_signals=tuple(
                    moderate_signals
                ),

                strong_signals=tuple(
                    strong_signals
                ),

                warnings=tuple(
                    warnings
                ),

                pages=tuple(
                    final_pages
                ),
            )

        finally:

            document.close()

    # ========================================================
    # Analyze Page
    # ========================================================

    def _analyze_page(
        self,
        *,
        document: fitz.Document,
        page: fitz.Page,
        page_number: int,
    ) -> dict:

        # ----------------------------------------------------
        # Content streams
        # ----------------------------------------------------

        try:

            contents = (
                page.get_contents()
            )

            if isinstance(
                contents,
                int,
            ):

                contents = [
                    contents
                ]

            contents = (
                contents or []
            )

        except Exception:

            contents = []

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        try:

            images = (
                page.get_images(
                    full=True
                )
            )

        except Exception:

            images = []

        # ----------------------------------------------------
        # Fonts
        # ----------------------------------------------------

        try:

            fonts = (
                page.get_fonts(
                    full=True
                )
            )

        except Exception:

            fonts = []

        # ----------------------------------------------------
        # XObjects
        # ----------------------------------------------------

        try:

            xobjects = (
                page.get_xobjects()
            )

        except Exception:

            xobjects = []

        # ----------------------------------------------------
        # Drawings
        # ----------------------------------------------------

        try:

            drawings = (
                page.get_drawings()
            )

        except Exception:

            drawings = []

        # ----------------------------------------------------
        # Links
        # ----------------------------------------------------

        try:

            links = (
                page.get_links()
            )

        except Exception:

            links = []

        # ----------------------------------------------------
        # Annotations
        # ----------------------------------------------------

        annotation_count = 0

        try:

            annotation = (
                page.first_annot
            )

            while annotation is not None:

                annotation_count += 1

                annotation = (
                    annotation.next
                )

        except Exception:

            annotation_count = 0

        # ----------------------------------------------------
        # Resource signature
        # ----------------------------------------------------

        font_names = []

        for font in fonts:

            try:

                # PyMuPDF font tuples contain several fields.
                # We only need a stable generic representation.

                font_names.append(
                    "|".join(
                        str(value)
                        for value
                        in font[1:5]
                    )
                )

            except Exception:

                continue

        image_types = []

        for image in images:

            try:

                image_types.append(
                    str(
                        image[2:5]
                    )
                )

            except Exception:

                continue

        signature_parts = [

            "streams="
            + str(
                len(contents)
            ),

            "fonts="
            + ",".join(
                sorted(
                    font_names
                )
            ),

            "images="
            + ",".join(
                sorted(
                    image_types
                )
            ),

            "xobjects="
            + str(
                len(xobjects)
            ),
        ]

        resource_signature = (
            "||".join(
                signature_parts
            )
        )

        return {

            "page_number": (
                page_number
            ),

            "page_xref": int(
                page.xref
            ),

            "content_stream_count": len(
                contents
            ),

            "image_count": len(
                images
            ),

            "font_count": len(
                fonts
            ),

            "xobject_count": len(
                xobjects
            ),

            "annotation_count": (
                annotation_count
            ),

            "link_count": len(
                links
            ),

            "drawing_count": len(
                drawings
            ),

            "resource_signature": (
                resource_signature
            ),
        }

    # ========================================================
    # Relative Outlier
    # ========================================================

    def _relative_outlier(
        self,
        value: float,
        baseline: float,
    ) -> bool:

        if baseline <= 0:

            return value >= 2

        difference = abs(
            value
            - baseline
        )

        return (
            difference
            / baseline
        ) > self.RELATIVE_THRESHOLD

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

    # ========================================================
    # Page Formatter
    # ========================================================

    @staticmethod
    def _page_list(
        pages,
        limit: int = 15,
    ) -> str:

        pages = tuple(
            pages
        )

        visible = pages[
            :limit
        ]

        text = ", ".join(
            str(page)
            for page in visible
        )

        remaining = (
            len(pages)
            - len(visible)
        )

        if remaining > 0:

            text += (
                f" (+{remaining} more)"
            )

        return text


# ============================================================
# Default Instance
# ============================================================


pdf_object_profile_analyzer = (
    PDFObjectProfileAnalyzer()
)