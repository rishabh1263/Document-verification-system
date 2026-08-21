"""
Generic PDF Page Consistency Analyzer.

Compares pages inside a PDF to identify unusual page-level
structural variations.

Signals considered:
- text length
- text block count
- image count
- drawing count
- font families
- font sizes
- page dimensions
- text density

Important:
- No OCR.
- No bank-specific templates.
- Outliers are evidence, not proof of tampering.
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
class PageEvidence:
    page_number: int

    text_length: int
    text_block_count: int

    image_count: int
    drawing_count: int

    fonts: tuple[str, ...]
    font_sizes: tuple[float, ...]

    width: float
    height: float

    text_density: float


# ============================================================
# Result
# ============================================================


@dataclass(frozen=True)
class PageConsistencyResult:
    page_count: int
    pages_analyzed: int

    dominant_fonts: tuple[str, ...]

    median_text_length: float
    median_text_block_count: float
    median_image_count: float
    median_drawing_count: float
    median_text_density: float

    font_outlier_pages: tuple[int, ...]
    font_size_outlier_pages: tuple[int, ...]

    text_length_outlier_pages: tuple[int, ...]
    text_block_outlier_pages: tuple[int, ...]

    image_outlier_pages: tuple[int, ...]
    drawing_outlier_pages: tuple[int, ...]

    dimension_outlier_pages: tuple[int, ...]
    density_outlier_pages: tuple[int, ...]

    structural_outlier_pages: tuple[int, ...]

    consistency_score: float

    pages: tuple[PageEvidence, ...]

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Analyzer
# ============================================================


class PageConsistencyAnalyzer:

    # Relative deviation thresholds

    TEXT_LENGTH_THRESHOLD = 0.60
    TEXT_BLOCK_THRESHOLD = 0.50
    DRAWING_THRESHOLD = 0.60
    DENSITY_THRESHOLD = 0.60

    FONT_SIZE_TOLERANCE = 2.5
    DIMENSION_TOLERANCE = 5.0

    # A page becomes a combined structural outlier when
    # multiple independent anomaly categories occur.

    STRUCTURAL_SIGNAL_THRESHOLD = 2

    # ========================================================
    # Public API
    # ========================================================

    def analyze(
        self,
        file_bytes: bytes,
    ) -> PageConsistencyResult:

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
                "Unable to open PDF for page consistency "
                "analysis."
            ) from exc

        try:

            if document.needs_pass:

                raise ValueError(
                    "Password-protected PDF cannot be "
                    "analyzed for page consistency."
                )

            if document.page_count == 0:

                raise ValueError(
                    "PDF contains no pages."
                )

            # =================================================
            # Collect page evidence
            # =================================================

            pages = tuple(
                self._analyze_page(
                    document[index],
                    index + 1,
                )
                for index in range(
                    document.page_count
                )
            )

            page_count = len(
                pages
            )

            # =================================================
            # Baselines
            # =================================================

            median_text_length = median(
                p.text_length
                for p in pages
            )

            median_text_blocks = median(
                p.text_block_count
                for p in pages
            )

            median_images = median(
                p.image_count
                for p in pages
            )

            median_drawings = median(
                p.drawing_count
                for p in pages
            )

            median_density = median(
                p.text_density
                for p in pages
            )

            dominant_fonts = (
                self._dominant_fonts(
                    pages
                )
            )

            dominant_font_sizes = (
                self._dominant_font_sizes(
                    pages
                )
            )

            baseline_width = median(
                p.width
                for p in pages
            )

            baseline_height = median(
                p.height
                for p in pages
            )

            # =================================================
            # Outlier detection
            # =================================================

            font_outliers: list[int] = []
            font_size_outliers: list[int] = []

            text_length_outliers: list[int] = []
            text_block_outliers: list[int] = []

            image_outliers: list[int] = []
            drawing_outliers: list[int] = []

            dimension_outliers: list[int] = []
            density_outliers: list[int] = []

            # Page → number of independent anomaly categories

            anomaly_counts: dict[int, int] = {
                p.page_number: 0
                for p in pages
            }

            for page in pages:

                # --------------------------------------------
                # Font-family anomaly
                # --------------------------------------------

                unexpected_fonts = (
                    set(page.fonts)
                    - set(dominant_fonts)
                )

                if unexpected_fonts:

                    font_outliers.append(
                        page.page_number
                    )

                    anomaly_counts[
                        page.page_number
                    ] += 1

                # --------------------------------------------
                # Font-size anomaly
                # --------------------------------------------

                if self._font_size_outlier(
                    page.font_sizes,
                    dominant_font_sizes,
                ):

                    font_size_outliers.append(
                        page.page_number
                    )

                    anomaly_counts[
                        page.page_number
                    ] += 1

                # --------------------------------------------
                # Text-length anomaly
                # --------------------------------------------

                if self._relative_outlier(
                    page.text_length,
                    median_text_length,
                    self.TEXT_LENGTH_THRESHOLD,
                ):

                    text_length_outliers.append(
                        page.page_number
                    )

                    anomaly_counts[
                        page.page_number
                    ] += 1

                # --------------------------------------------
                # Text-block anomaly
                # --------------------------------------------

                if self._relative_outlier(
                    page.text_block_count,
                    median_text_blocks,
                    self.TEXT_BLOCK_THRESHOLD,
                ):

                    text_block_outliers.append(
                        page.page_number
                    )

                    anomaly_counts[
                        page.page_number
                    ] += 1

                # --------------------------------------------
                # Image anomaly
                # --------------------------------------------

                if self._count_outlier(
                    page.image_count,
                    median_images,
                ):

                    image_outliers.append(
                        page.page_number
                    )

                    anomaly_counts[
                        page.page_number
                    ] += 1

                # --------------------------------------------
                # Drawing anomaly
                # --------------------------------------------

                if self._relative_outlier(
                    page.drawing_count,
                    median_drawings,
                    self.DRAWING_THRESHOLD,
                ):

                    drawing_outliers.append(
                        page.page_number
                    )

                    anomaly_counts[
                        page.page_number
                    ] += 1

                # --------------------------------------------
                # Page dimension anomaly
                # --------------------------------------------

                if (
                    abs(
                        page.width
                        - baseline_width
                    )
                    > self.DIMENSION_TOLERANCE
                    or
                    abs(
                        page.height
                        - baseline_height
                    )
                    > self.DIMENSION_TOLERANCE
                ):

                    dimension_outliers.append(
                        page.page_number
                    )

                    anomaly_counts[
                        page.page_number
                    ] += 1

                # --------------------------------------------
                # Text-density anomaly
                # --------------------------------------------

                if self._relative_outlier(
                    page.text_density,
                    median_density,
                    self.DENSITY_THRESHOLD,
                ):

                    density_outliers.append(
                        page.page_number
                    )

                    anomaly_counts[
                        page.page_number
                    ] += 1

            # =================================================
            # Combined structural outliers
            # =================================================

            structural_outliers = tuple(
                page_number
                for page_number, count
                in anomaly_counts.items()
                if count
                >= self.STRUCTURAL_SIGNAL_THRESHOLD
            )

            # =================================================
            # Consistency score
            # =================================================

            consistency_score = (
                (
                    page_count
                    - len(
                        structural_outliers
                    )
                )
                / page_count
            ) * 100

            return PageConsistencyResult(

                page_count=page_count,

                pages_analyzed=page_count,

                dominant_fonts=(
                    dominant_fonts
                ),

                median_text_length=float(
                    median_text_length
                ),

                median_text_block_count=float(
                    median_text_blocks
                ),

                median_image_count=float(
                    median_images
                ),

                median_drawing_count=float(
                    median_drawings
                ),

                median_text_density=float(
                    median_density
                ),

                font_outlier_pages=tuple(
                    font_outliers
                ),

                font_size_outlier_pages=tuple(
                    font_size_outliers
                ),

                text_length_outlier_pages=tuple(
                    text_length_outliers
                ),

                text_block_outlier_pages=tuple(
                    text_block_outliers
                ),

                image_outlier_pages=tuple(
                    image_outliers
                ),

                drawing_outlier_pages=tuple(
                    drawing_outliers
                ),

                dimension_outlier_pages=tuple(
                    dimension_outliers
                ),

                density_outlier_pages=tuple(
                    density_outliers
                ),

                structural_outlier_pages=(
                    structural_outliers
                ),

                consistency_score=round(
                    consistency_score,
                    2,
                ),

                pages=pages,
            )

        finally:

            document.close()

    # ========================================================
    # Individual Page Analysis
    # ========================================================

    @staticmethod
    def _analyze_page(
        page: fitz.Page,
        page_number: int,
    ) -> PageEvidence:

        text = page.get_text(
            "text"
        ) or ""

        blocks = page.get_text(
            "blocks"
        )

        images = page.get_images(
            full=True
        )

        try:
            drawings = page.get_drawings()
        except Exception:
            drawings = []

        # ====================================================
        # Font extraction
        # ====================================================

        fonts: set[str] = set()
        font_sizes: set[float] = set()

        try:

            text_dict = page.get_text(
                "dict"
            )

            for block in text_dict.get(
                "blocks",
                []
            ):

                if block.get(
                    "type"
                ) != 0:
                    continue

                for line in block.get(
                    "lines",
                    []
                ):

                    for span in line.get(
                        "spans",
                        []
                    ):

                        font = span.get(
                            "font"
                        )

                        if font:

                            fonts.add(
                                str(font)
                            )

                        size = span.get(
                            "size"
                        )

                        if size:

                            font_sizes.add(
                                round(
                                    float(size),
                                    1,
                                )
                            )

        except Exception:
            pass

        # ====================================================
        # Page geometry
        # ====================================================

        width = float(
            page.rect.width
        )

        height = float(
            page.rect.height
        )

        area = max(
            width * height,
            1.0,
        )

        text_density = (
            len(text)
            / area
        )

        return PageEvidence(

            page_number=page_number,

            text_length=len(
                text.strip()
            ),

            text_block_count=len(
                blocks
            ),

            image_count=len(
                images
            ),

            drawing_count=len(
                drawings
            ),

            fonts=tuple(
                sorted(
                    fonts
                )
            ),

            font_sizes=tuple(
                sorted(
                    font_sizes
                )
            ),

            width=round(
                width,
                2,
            ),

            height=round(
                height,
                2,
            ),

            text_density=round(
                text_density,
                8,
            ),
        )

    # ========================================================
    # Dominant Fonts
    # ========================================================

    @staticmethod
    def _dominant_fonts(
        pages: tuple[PageEvidence, ...],
    ) -> tuple[str, ...]:

        counts: dict[str, int] = {}

        for page in pages:

            for font in set(
                page.fonts
            ):

                counts[font] = (
                    counts.get(
                        font,
                        0,
                    )
                    + 1
                )

        minimum_pages = max(
            2,
            int(
                len(pages)
                * 0.50
            ),
        )

        dominant = [
            font
            for font, count
            in counts.items()
            if count >= minimum_pages
        ]

        return tuple(
            sorted(
                dominant
            )
        )

    # ========================================================
    # Dominant Font Sizes
    # ========================================================

    @staticmethod
    def _dominant_font_sizes(
        pages: tuple[PageEvidence, ...],
    ) -> tuple[float, ...]:

        counts: dict[float, int] = {}

        for page in pages:

            for size in set(
                page.font_sizes
            ):

                counts[size] = (
                    counts.get(
                        size,
                        0,
                    )
                    + 1
                )

        minimum_pages = max(
            2,
            int(
                len(pages)
                * 0.50
            ),
        )

        dominant = [
            size
            for size, count
            in counts.items()
            if count >= minimum_pages
        ]

        return tuple(
            sorted(
                dominant
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

        ratio = (
            difference
            / baseline
        )

        return (
            ratio > threshold
        )

    # ========================================================
    # Count Outlier
    # ========================================================

    @staticmethod
    def _count_outlier(
        value: int,
        baseline: float,
    ) -> bool:

        # Most pages have no images but one page suddenly does.

        if baseline == 0:

            return value >= 2

        return (
            value
            > baseline * 3
        )

    # ========================================================
    # Font Size Outlier
    # ========================================================

    def _font_size_outlier(
        self,
        page_sizes: tuple[float, ...],
        dominant_sizes: tuple[float, ...],
    ) -> bool:

        if not page_sizes:
            return False

        if not dominant_sizes:
            return False

        for page_size in page_sizes:

            closest = min(
                abs(
                    page_size
                    - dominant
                )
                for dominant
                in dominant_sizes
            )

            if (
                closest
                > self.FONT_SIZE_TOLERANCE
            ):
                return True

        return False


# ============================================================
# Default Instance
# ============================================================


page_consistency_analyzer = (
    PageConsistencyAnalyzer()
)