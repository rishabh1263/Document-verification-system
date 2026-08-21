"""
Generic PDF Integrity Analyzer.

Collects metadata and structural evidence from a PDF.

Responsibilities:
- inspect basic PDF structure
- collect PDF metadata
- detect encryption/password requirements
- count annotations
- count embedded files
- inspect image usage
- inspect font usage
- inspect page-level structural characteristics
- collect PDF xref/object information
- report whether useful descriptive metadata exists

Important:
This module does NOT:
- perform OCR
- decide whether the document is a bank statement
- declare a document fake or genuine
- calculate the final tamper/fraud score
- extract transactions
- perform loan eligibility logic

This analyzer collects FACTS only.
Risk interpretation belongs to TamperDetector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO

import fitz


# ============================================================
# Result Model
# ============================================================


@dataclass(frozen=True)
class PDFIntegrityResult:
    """
    Raw integrity evidence collected from a PDF.

    These values are factual observations only.

    No individual field should automatically be interpreted
    as proof that a document has been tampered with.
    """

    # --------------------------------------------------------
    # Basic PDF information
    # --------------------------------------------------------

    page_count: int
    pdf_format: str

    encrypted: bool
    needs_password: bool

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    title: str
    author: str
    subject: str
    creator: str
    producer: str

    creation_date: str
    modification_date: str

    metadata_available: bool

    # --------------------------------------------------------
    # PDF features
    # --------------------------------------------------------

    annotation_count: int
    pages_with_annotations: int

    embedded_file_count: int

    # --------------------------------------------------------
    # Image structure
    # --------------------------------------------------------

    total_images: int
    pages_with_images: int

    # --------------------------------------------------------
    # Font structure
    # --------------------------------------------------------

    font_count: int
    unique_fonts: tuple[str, ...]

    # --------------------------------------------------------
    # Page structure
    # --------------------------------------------------------

    pages_with_text: int
    pages_without_text: int

    # --------------------------------------------------------
    # Internal PDF structure
    # --------------------------------------------------------

    xref_length: int

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Analyzer
# ============================================================


class PDFIntegrityAnalyzer:
    """
    Inspect metadata and structural characteristics of a PDF.

    This class deliberately separates evidence collection from
    tamper-risk interpretation.

    Example:

        PDFIntegrityAnalyzer
                ↓
        PDFIntegrityResult
                ↓
        TamperDetector

    This makes the integrity layer modular and replaceable.
    """

    # ========================================================
    # Public API
    # ========================================================

    def analyze(
        self,
        file_bytes: bytes,
    ) -> PDFIntegrityResult:

        if not file_bytes:
            raise ValueError(
                "PDF bytes are required."
            )

        # ----------------------------------------------------
        # Open PDF
        # ----------------------------------------------------

        try:
            document = fitz.open(
                stream=BytesIO(file_bytes),
                filetype="pdf",
            )

        except Exception as exc:
            raise ValueError(
                "Unable to open PDF for integrity analysis."
            ) from exc

        try:

            # ------------------------------------------------
            # Password handling
            # ------------------------------------------------

            if document.needs_pass:
                return self._build_password_protected_result(
                    document
                )

            metadata = document.metadata or {}

            # ------------------------------------------------
            # Collect structural evidence
            # ------------------------------------------------

            annotation_count, pages_with_annotations = (
                self._analyze_annotations(
                    document
                )
            )

            embedded_file_count = (
                self._count_embedded_files(
                    document
                )
            )

            total_images, pages_with_images = (
                self._analyze_images(
                    document
                )
            )

            unique_fonts = self._analyze_fonts(
                document
            )

            pages_with_text, pages_without_text = (
                self._analyze_page_text(
                    document
                )
            )

            xref_length = self._get_xref_length(
                document
            )

            metadata_available = (
                self._has_useful_metadata(
                    metadata
                )
            )

            # ------------------------------------------------
            # Build immutable result
            # ------------------------------------------------

            return PDFIntegrityResult(

                # Basic PDF information

                page_count=document.page_count,

                pdf_format=self._safe_value(
                    metadata.get("format")
                ),

                encrypted=bool(
                    document.is_encrypted
                ),

                needs_password=bool(
                    document.needs_pass
                ),

                # Metadata

                title=self._safe_value(
                    metadata.get("title")
                ),

                author=self._safe_value(
                    metadata.get("author")
                ),

                subject=self._safe_value(
                    metadata.get("subject")
                ),

                creator=self._safe_value(
                    metadata.get("creator")
                ),

                producer=self._safe_value(
                    metadata.get("producer")
                ),

                creation_date=self._safe_value(
                    metadata.get("creationDate")
                ),

                modification_date=self._safe_value(
                    metadata.get("modDate")
                ),

                metadata_available=metadata_available,

                # PDF features

                annotation_count=annotation_count,

                pages_with_annotations=(
                    pages_with_annotations
                ),

                embedded_file_count=(
                    embedded_file_count
                ),

                # Images

                total_images=total_images,

                pages_with_images=pages_with_images,

                # Fonts

                font_count=len(
                    unique_fonts
                ),

                unique_fonts=unique_fonts,

                # Page text

                pages_with_text=pages_with_text,

                pages_without_text=pages_without_text,

                # Internal structure

                xref_length=xref_length,
            )

        finally:
            document.close()

    # ========================================================
    # Annotation Analysis
    # ========================================================

    @staticmethod
    def _analyze_annotations(
        document: fitz.Document,
    ) -> tuple[int, int]:
        """
        Count annotations and pages containing annotations.

        Annotations are not automatically suspicious.

        Examples of legitimate annotations include:
        - hyperlinks
        - comments
        - form-related elements
        """

        annotation_count = 0
        pages_with_annotations = 0

        for page in document:

            annotations = page.annots()

            if annotations is None:
                continue

            page_annotation_count = sum(
                1
                for _ in annotations
            )

            if page_annotation_count > 0:

                annotation_count += (
                    page_annotation_count
                )

                pages_with_annotations += 1

        return (
            annotation_count,
            pages_with_annotations,
        )

    # ========================================================
    # Embedded File Analysis
    # ========================================================

    @staticmethod
    def _count_embedded_files(
        document: fitz.Document,
    ) -> int:
        """
        Count files embedded inside the PDF.

        Embedded files are collected as evidence only.
        """

        try:
            return int(
                document.embfile_count()
            )

        except Exception:
            return 0

    # ========================================================
    # Image Analysis
    # ========================================================

    @staticmethod
    def _analyze_images(
        document: fitz.Document,
    ) -> tuple[int, int]:
        """
        Count image references across the document.

        Also count how many pages contain at least one image.

        Images are common in legitimate bank statements:
        - logos
        - signatures
        - QR codes
        - scanned pages

        Therefore image presence alone is NOT suspicious.
        """

        total_images = 0
        pages_with_images = 0

        for page in document:

            try:
                images = page.get_images(
                    full=True
                )

            except Exception:
                images = []

            image_count = len(images)

            total_images += image_count

            if image_count > 0:
                pages_with_images += 1

        return (
            total_images,
            pages_with_images,
        )

    # ========================================================
    # Font Analysis
    # ========================================================

    @staticmethod
    def _analyze_fonts(
        document: fitz.Document,
    ) -> tuple[str, ...]:
        """
        Collect unique font names used across the PDF.

        Different fonts are not automatically suspicious.

        This information can later help TamperDetector identify
        unusual one-off font changes or inconsistent document
        construction.
        """

        fonts: set[str] = set()

        for page in document:

            try:
                page_fonts = page.get_fonts(
                    full=True
                )

            except Exception:
                continue

            for font in page_fonts:

                # PyMuPDF font tuples can vary slightly
                # depending on version.
                #
                # Typical structure includes:
                #
                # (
                #   xref,
                #   ext,
                #   type,
                #   basefont,
                #   name,
                #   encoding,
                #   ...
                # )

                font_name = ""

                if len(font) > 3:
                    font_name = str(
                        font[3]
                    ).strip()

                if font_name:
                    fonts.add(
                        font_name
                    )

        return tuple(
            sorted(fonts)
        )

    # ========================================================
    # Page Text Structure
    # ========================================================

    @staticmethod
    def _analyze_page_text(
        document: fitz.Document,
    ) -> tuple[int, int]:
        """
        Determine how many pages contain embedded text.

        This does NOT perform OCR.

        A page containing only an image may therefore appear
        as a page without embedded text.
        """

        pages_with_text = 0
        pages_without_text = 0

        for page in document:

            try:
                text = page.get_text(
                    "text"
                ).strip()

            except Exception:
                text = ""

            if text:
                pages_with_text += 1

            else:
                pages_without_text += 1

        return (
            pages_with_text,
            pages_without_text,
        )

    # ========================================================
    # XREF / Object Structure
    # ========================================================

    @staticmethod
    def _get_xref_length(
        document: fitz.Document,
    ) -> int:
        """
        Return the number of entries in the PDF cross-reference
        structure.

        XREF information is useful structural evidence.

        A large xref count is NOT automatically suspicious.
        Complex legitimate PDFs can contain many objects.
        """

        try:
            return int(
                document.xref_length()
            )

        except Exception:
            return 0

    # ========================================================
    # Metadata Analysis
    # ========================================================

    @staticmethod
    def _has_useful_metadata(
        metadata: dict,
    ) -> bool:
        """
        Determine whether useful descriptive metadata exists.

        PDF format/version alone does not count.

        Example:

            format = "PDF 1.4"

        while creator, producer, dates, etc. are empty should
        produce:

            metadata_available = False
        """

        useful_keys = (
            "title",
            "author",
            "subject",
            "creator",
            "producer",
            "creationDate",
            "modDate",
        )

        return any(
            bool(
                PDFIntegrityAnalyzer._safe_value(
                    metadata.get(key)
                )
            )
            for key in useful_keys
        )

    # ========================================================
    # Password-Protected PDF
    # ========================================================

    def _build_password_protected_result(
        self,
        document: fitz.Document,
    ) -> PDFIntegrityResult:
        """
        Build a safe partial result when the PDF requires a
        password.

        We should not pretend structural analysis completed
        successfully when content is inaccessible.
        """

        metadata = document.metadata or {}

        return PDFIntegrityResult(

            page_count=document.page_count,

            pdf_format=self._safe_value(
                metadata.get("format")
            ),

            encrypted=bool(
                document.is_encrypted
            ),

            needs_password=True,

            title=self._safe_value(
                metadata.get("title")
            ),

            author=self._safe_value(
                metadata.get("author")
            ),

            subject=self._safe_value(
                metadata.get("subject")
            ),

            creator=self._safe_value(
                metadata.get("creator")
            ),

            producer=self._safe_value(
                metadata.get("producer")
            ),

            creation_date=self._safe_value(
                metadata.get("creationDate")
            ),

            modification_date=self._safe_value(
                metadata.get("modDate")
            ),

            metadata_available=(
                self._has_useful_metadata(
                    metadata
                )
            ),

            annotation_count=0,

            pages_with_annotations=0,

            embedded_file_count=0,

            total_images=0,

            pages_with_images=0,

            font_count=0,

            unique_fonts=(),

            pages_with_text=0,

            pages_without_text=0,

            xref_length=self._get_xref_length(
                document
            ),
        )

    # ========================================================
    # Generic Helpers
    # ========================================================

    @staticmethod
    def _safe_value(
        value,
    ) -> str:
        """
        Convert optional metadata values into clean strings.
        """

        if value is None:
            return ""

        return str(value).strip()


# ============================================================
# Default Reusable Instance
# ============================================================

pdf_integrity_analyzer = PDFIntegrityAnalyzer()