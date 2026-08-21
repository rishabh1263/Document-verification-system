"""
Production-oriented PDF integrity analysis for ITR documents.

Purpose
-------
Analyze the technical structure of an uploaded PDF and produce
explainable authenticity signals.

Important
---------
This module does NOT determine whether an ITR is legally genuine.

It identifies technical anomalies that may indicate:
    - editing
    - reconstruction
    - suspicious PDF generation
    - embedded active content
    - unusual document structure
    - metadata associated with editing software

A single weak signal must never be treated as proof of fraud.

Architecture
------------

    PDF bytes
       |
       +--> SHA-256
       |
       +--> metadata
       |
       +--> structure
       |
       +--> pages
       |
       +--> fonts
       |
       +--> images
       |
       +--> annotations
       |
       +--> embedded files
       |
       +--> JavaScript/actions
       |
       +--> text/image analysis
       |
       v
    IntegrityResult
       |
       +--> findings
       +--> risk score
       +--> risk level
       +--> explainable reason
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import fitz


logger = logging.getLogger(__name__)


# ==========================================================
# ENUMS
# ==========================================================


class IntegritySeverity(str, Enum):
    """
    Severity of a PDF integrity finding.
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IntegrityStatus(str, Enum):
    """
    Overall technical integrity status.
    """

    CLEAN = "clean"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"


# ==========================================================
# DATA MODELS
# ==========================================================


@dataclass(frozen=True)
class IntegrityFinding:
    """
    One explainable PDF integrity finding.
    """

    rule_id: str

    category: str

    severity: IntegritySeverity

    message: str

    reason: str

    evidence: dict[str, Any] = field(
        default_factory=dict
    )

    score: float = 0.0

    @property
    def severity_rank(self) -> int:
        """
        Numeric ordering for severity comparison.
        """

        return {
            IntegritySeverity.INFO: 0,
            IntegritySeverity.LOW: 1,
            IntegritySeverity.MEDIUM: 2,
            IntegritySeverity.HIGH: 3,
            IntegritySeverity.CRITICAL: 4,
        }[self.severity]


@dataclass(frozen=True)
class PageIntegrity:
    """
    Technical information about one PDF page.
    """

    page_number: int

    text_characters: int

    word_count: int

    image_count: int

    drawing_count: int

    block_count: int

    has_text: bool

    has_images: bool

    image_only: bool


@dataclass(frozen=True)
class PDFIntegrityResult:
    """
    Complete technical PDF integrity result.
    """

    success: bool

    status: IntegrityStatus

    risk_level: IntegritySeverity

    risk_score: float

    confidence: float

    document_hash: str | None

    file_size: int

    page_count: int

    encrypted: bool

    metadata: dict[str, Any]

    pages: tuple[PageIntegrity, ...]

    findings: tuple[IntegrityFinding, ...]

    reason: str

    summary: str

    @property
    def has_high_risk_finding(self) -> bool:
        return any(
            finding.severity
            in {
                IntegritySeverity.HIGH,
                IntegritySeverity.CRITICAL,
            }
            for finding in self.findings
        )

    @property
    def has_critical_finding(self) -> bool:
        return any(
            finding.severity
            == IntegritySeverity.CRITICAL
            for finding in self.findings
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize result for API responses.
        """

        return {
            "success": self.success,

            "status": self.status.value,

            "risk_level": (
                self.risk_level.value
            ),

            "risk_score": self.risk_score,

            "confidence": self.confidence,

            "document_hash": self.document_hash,

            "file_size": self.file_size,

            "page_count": self.page_count,

            "encrypted": self.encrypted,

            "metadata": self.metadata,

            "pages": [
                asdict(page)
                for page in self.pages
            ],

            "findings": [
                {
                    "rule_id": finding.rule_id,

                    "category": finding.category,

                    "severity": (
                        finding.severity.value
                    ),

                    "message": finding.message,

                    "reason": finding.reason,

                    "evidence": finding.evidence,

                    "score": finding.score,
                }
                for finding in self.findings
            ],

            "reason": self.reason,

            "summary": self.summary,
        }


# ==========================================================
# ANALYZER
# ==========================================================


class PDFIntegrityAnalyzer:
    """
    Analyze the technical integrity of a PDF.

    The analyzer is deliberately conservative.

    It detects signals; it does not declare legal authenticity.
    """

    # ------------------------------------------------------
    # Suspicious PDF editor names
    # ------------------------------------------------------

    EDITOR_TERMS = (
        "photoshop",
        "adobe photoshop",
        "gimp",
        "canva",
        "illustrator",
        "coreldraw",
        "paint",
        "paint.net",
        "nitro pro",
        "foxit",
        "pdfelement",
        "pdf-xchange",
        "master pdf editor",
    )

    # ------------------------------------------------------
    # Common PDF generators that are NOT inherently suspicious
    # ------------------------------------------------------

    NORMAL_PRODUCERS = (
        "itext",
        "reportlab",
        "mupdf",
        "pymupdf",
        "adobe",
        "wkhtmltopdf",
        "pdfkit",
        "weasyprint",
    )

    # ======================================================
    # PUBLIC API
    # ======================================================

    def analyze(
        self,
        pdf_bytes: bytes,
    ) -> PDFIntegrityResult:
        """
        Analyze PDF bytes.

        Args:
            pdf_bytes:
                Complete uploaded PDF bytes.

        Returns:
            PDFIntegrityResult
        """

        if not pdf_bytes:
            return self._empty_result()

        document_hash = hashlib.sha256(
            pdf_bytes
        ).hexdigest()

        file_size = len(
            pdf_bytes
        )

        findings: list[
            IntegrityFinding
        ] = []

        pages: list[
            PageIntegrity
        ] = []

        try:
            document = fitz.open(
                stream=pdf_bytes,
                filetype="pdf",
            )
        except Exception as exc:
            logger.exception(
                "Unable to open PDF for integrity analysis."
            )

            finding = IntegrityFinding(
                rule_id="PDF_OPEN_FAILED",

                category="document",

                severity=(
                    IntegritySeverity.CRITICAL
                ),

                message=(
                    "The uploaded file could not be "
                    "opened as a valid PDF."
                ),

                reason=(
                    "The PDF parser failed to open the uploaded "
                    "document. The file may be corrupted, malformed, "
                    "encrypted in an unsupported way, or not actually "
                    "be a valid PDF."
                ),

                evidence={
                    "error": str(exc),
                },

                score=100.0,
            )

            return PDFIntegrityResult(
                success=False,

                status=(
                    IntegrityStatus.HIGH_RISK
                ),

                risk_level=(
                    IntegritySeverity.CRITICAL
                ),

                risk_score=1.0,

                confidence=0.99,

                document_hash=document_hash,

                file_size=file_size,

                page_count=0,

                encrypted=False,

                metadata={},

                pages=(),

                findings=(finding,),

                reason=(
                    finding.reason
                ),

                summary=(
                    "PDF integrity analysis failed."
                ),
            )

        try:
            # ==============================================
            # METADATA
            # ==============================================

            metadata = self._metadata(
                document
            )

            findings.extend(
                self._check_metadata(
                    metadata
                )
            )

            # ==============================================
            # ENCRYPTION
            # ==============================================

            if document.is_encrypted:

                findings.append(
                    IntegrityFinding(
                        rule_id=(
                            "PDF_ENCRYPTED"
                        ),

                        category="security",

                        severity=(
                            IntegritySeverity.MEDIUM
                        ),

                        message=(
                            "The PDF is encrypted."
                        ),

                        reason=(
                            "Encrypted PDFs can legitimately be "
                            "used for document protection, so encryption "
                            "is not proof of manipulation. It does, "
                            "however, limit structural inspection."
                        ),

                        evidence={
                            "encrypted": True,
                        },

                        score=20.0,
                    )
                )

            # ==============================================
            # JAVASCRIPT
            # ==============================================

            findings.extend(
                self._check_javascript(
                    document
                )
            )

            # ==============================================
            # EMBEDDED FILES
            # ==============================================

            findings.extend(
                self._check_embedded_files(
                    document
                )
            )

            # ==============================================
            # DOCUMENT LINKS / ACTIONS
            # ==============================================

            findings.extend(
                self._check_document_actions(
                    document
                )
            )

            # ==============================================
            # PAGE ANALYSIS
            # ==============================================

            for index in range(
                document.page_count
            ):

                page = document.load_page(
                    index
                )

                page_info = (
                    self._analyze_page(
                        page,
                        index + 1,
                    )
                )

                pages.append(
                    page_info
                )

                findings.extend(
                    self._check_page(
                        page_info
                    )
                )

            # ==============================================
            # FONT ANALYSIS
            # ==============================================

            findings.extend(
                self._check_fonts(
                    document
                )
            )

            # ==============================================
            # OBJECT / STRUCTURE ANALYSIS
            # ==============================================

            findings.extend(
                self._check_structure(
                    document
                )
            )

            # ==============================================
            # MIXED CONTENT ANALYSIS
            # ==============================================

            findings.extend(
                self._check_mixed_content(
                    pages
                )
            )

            # ==============================================
            # RISK
            # ==============================================

            risk_score = (
                self._calculate_risk(
                    findings
                )
            )

            risk_level = (
                self._risk_level(
                    findings,
                    risk_score,
                )
            )

            status = (
                self._status(
                    risk_level
                )
            )

            reason = (
                self._build_reason(
                    findings,
                    status,
                )
            )

            summary = (
                self._build_summary(
                    findings,
                    status,
                )
            )

            confidence = (
                self._confidence(
                    findings,
                    risk_level,
                )
            )

            return PDFIntegrityResult(
                success=True,

                status=status,

                risk_level=risk_level,

                risk_score=risk_score,

                confidence=confidence,

                document_hash=document_hash,

                file_size=file_size,

                page_count=document.page_count,

                encrypted=document.is_encrypted,

                metadata=metadata,

                pages=tuple(
                    pages
                ),

                findings=tuple(
                    findings
                ),

                reason=reason,

                summary=summary,
            )

        finally:
            document.close()

    # ======================================================
    # EMPTY RESULT
    # ======================================================

    @staticmethod
    def _empty_result() -> PDFIntegrityResult:
        finding = IntegrityFinding(
            rule_id="PDF_EMPTY",

            category="document",

            severity=(
                IntegritySeverity.CRITICAL
            ),

            message=(
                "The uploaded PDF is empty."
            ),

            reason=(
                "No PDF bytes were supplied to the integrity "
                "analyzer."
            ),

            evidence={},

            score=100.0,
        )

        return PDFIntegrityResult(
            success=False,

            status=IntegrityStatus.HIGH_RISK,

            risk_level=(
                IntegritySeverity.CRITICAL
            ),

            risk_score=1.0,

            confidence=1.0,

            document_hash=None,

            file_size=0,

            page_count=0,

            encrypted=False,

            metadata={},

            pages=(),

            findings=(finding,),

            reason=finding.reason,

            summary=(
                "PDF integrity analysis could not start."
            ),
        )

    # ======================================================
    # METADATA
    # ======================================================

    @staticmethod
    def _metadata(
        document: fitz.Document,
    ) -> dict[str, Any]:
        """
        Return normalized PDF metadata.
        """

        raw = (
            document.metadata
            or {}
        )

        return {
            "format": raw.get(
                "format"
            ),

            "title": raw.get(
                "title"
            ),

            "author": raw.get(
                "author"
            ),

            "subject": raw.get(
                "subject"
            ),

            "keywords": raw.get(
                "keywords"
            ),

            "creator": raw.get(
                "creator"
            ),

            "producer": raw.get(
                "producer"
            ),

            "creation_date": raw.get(
                "creationDate"
            ),

            "modification_date": raw.get(
                "modDate"
            ),

            "encryption": raw.get(
                "encryption"
            ),
        }

    # ======================================================
    # METADATA RULES
    # ======================================================

    def _check_metadata(
        self,
        metadata: dict[str, Any],
    ) -> list[IntegrityFinding]:
        findings: list[
            IntegrityFinding
        ] = []

        creator = str(
            metadata.get(
                "creator"
            )
            or ""
        )

        producer = str(
            metadata.get(
                "producer"
            )
            or ""
        )

        combined = (
            f"{creator} {producer}"
        ).casefold()

        matched_editors = [
            term
            for term
            in self.EDITOR_TERMS
            if term.casefold()
            in combined
        ]

        if matched_editors:

            findings.append(
                IntegrityFinding(
                    rule_id=(
                        "PDF_EDITOR_METADATA"
                    ),

                    category="metadata",

                    severity=(
                        IntegritySeverity.MEDIUM
                    ),

                    message=(
                        "PDF metadata references "
                        "document editing software."
                    ),

                    reason=(
                        "The creator or producer metadata contains "
                        "software commonly associated with document "
                        "or image editing. This is a tampering signal, "
                        "not standalone proof that the ITR is fake."
                    ),

                    evidence={
                        "creator": creator,

                        "producer": producer,

                        "matched_terms": (
                            matched_editors
                        ),
                    },

                    score=30.0,
                )
            )

        creation_date = metadata.get(
            "creation_date"
        )

        modification_date = metadata.get(
            "modification_date"
        )

        if (
            creation_date
            and
            modification_date
            and
            creation_date
            !=
            modification_date
        ):

            findings.append(
                IntegrityFinding(
                    rule_id=(
                        "PDF_METADATA_MODIFIED"
                    ),

                    category="metadata",

                    severity=(
                        IntegritySeverity.LOW
                    ),

                    message=(
                        "PDF creation and modification "
                        "timestamps differ."
                    ),

                    reason=(
                        "The PDF contains different creation and "
                        "modification timestamps. This can occur during "
                        "normal PDF processing, so it is only a weak "
                        "tampering signal."
                    ),

                    evidence={
                        "creation_date": (
                            creation_date
                        ),

                        "modification_date": (
                            modification_date
                        ),
                    },

                    score=10.0,
                )
            )

        return findings

    # ======================================================
    # JAVASCRIPT
    # ======================================================

    @staticmethod
    def _check_javascript(
        document: fitz.Document,
    ) -> list[IntegrityFinding]:
        """
        Detect PDF JavaScript.

        JavaScript is unusual for normal tax documents.

        It is therefore treated as HIGH risk, not automatically fake.
        """

        findings: list[
            IntegrityFinding
        ] = []

        try:
            scripts = (
                document.embfile_names()
            )
        except Exception:
            scripts = []

        # The above call is deliberately not used as proof of JS.
        # PyMuPDF's public APIs vary across versions.
        #
        # We additionally inspect PDF XML/object text where possible.

        try:
            xref_length = (
                document.xref_length()
            )

            js_matches = 0

            for xref in range(
                1,
                xref_length,
            ):

                try:
                    object_text = (
                        document.xref_object(
                            xref,
                            compressed=False,
                        )
                        or ""
                    )
                except Exception:
                    continue

                lowered = (
                    object_text.casefold()
                )

                if (
                    "/javascript"
                    in lowered
                    or
                    "/js "
                    in lowered
                    or
                    "/js\n"
                    in lowered
                    or
                    "/js\r"
                    in lowered
                ):

                    js_matches += 1

            if js_matches:

                findings.append(
                    IntegrityFinding(
                        rule_id=(
                            "PDF_JAVASCRIPT"
                        ),

                        category="active_content",

                        severity=(
                            IntegritySeverity.HIGH
                        ),

                        message=(
                            "JavaScript content was detected "
                            "inside the PDF."
                        ),

                        reason=(
                            "JavaScript is uncommon in ordinary ITR "
                            "documents and can be used to implement "
                            "dynamic or potentially malicious behavior. "
                            "Its presence should therefore trigger "
                            "additional review."
                        ),

                        evidence={
                            "javascript_objects": (
                                js_matches
                            ),
                        },

                        score=65.0,
                    )
                )

        except Exception as exc:
            logger.debug(
                "Unable to inspect PDF JavaScript: %s",
                exc,
            )

        return findings

    # ======================================================
    # EMBEDDED FILES
    # ======================================================

    @staticmethod
    def _check_embedded_files(
        document: fitz.Document,
    ) -> list[IntegrityFinding]:
        findings: list[
            IntegrityFinding
        ] = []

        try:
            names = (
                document.embfile_names()
            )
        except Exception:
            names = []

        if names:

            findings.append(
                IntegrityFinding(
                    rule_id=(
                        "PDF_EMBEDDED_FILES"
                    ),

                    category="embedded_content",

                    severity=(
                        IntegritySeverity.MEDIUM
                    ),

                    message=(
                        "The PDF contains embedded files."
                    ),

                    reason=(
                        "Embedded files are not normally required "
                        "for a standard ITR document. They can be "
                        "legitimate, but they increase the document's "
                        "complexity and should be reviewed."
                    ),

                    evidence={
                        "count": len(names),

                        "names": names[
                            :20
                        ],
                    },

                    score=25.0,
                )
            )

        return findings

    # ======================================================
    # DOCUMENT ACTIONS
    # ======================================================

    @staticmethod
    def _check_document_actions(
        document: fitz.Document,
    ) -> list[IntegrityFinding]:
        findings: list[
            IntegrityFinding
        ] = []

        try:
            catalog = (
                document.pdf_catalog()
            )

            catalog_text = (
                str(catalog)
                .casefold()
            )

            suspicious_tokens = (
                "/openaction",
                "/aa ",
                "/launch",
            )

            matches = [
                token
                for token
                in suspicious_tokens
                if token
                in catalog_text
            ]

            if matches:

                findings.append(
                    IntegrityFinding(
                        rule_id=(
                            "PDF_DOCUMENT_ACTION"
                        ),

                        category="active_content",

                        severity=(
                            IntegritySeverity.HIGH
                        ),

                        message=(
                            "The PDF catalog contains "
                            "active document actions."
                        ),

                        reason=(
                            "The PDF contains actions that can execute "
                            "or trigger behavior when the document is "
                            "opened or interacted with. Such behavior "
                            "is unusual for standard ITR documents."
                        ),

                        evidence={
                            "matches": matches,
                        },

                        score=60.0,
                    )
                )

        except Exception as exc:
            logger.debug(
                "Unable to inspect PDF catalog actions: %s",
                exc,
            )

        return findings

    # ======================================================
    # PAGE ANALYSIS
    # ======================================================

    @staticmethod
    def _analyze_page(
        page: fitz.Page,
        page_number: int,
    ) -> PageIntegrity:
        """
        Analyze page-level text/image structure.
        """

        text = (
            page.get_text(
                "text"
            )
            or ""
        )

        words = (
            page.get_text(
                "words"
            )
            or []
        )

        images = (
            page.get_images(
                full=True
            )
            or []
        )

        drawings = (
            page.get_drawings()
            or []
        )

        blocks = (
            page.get_text(
                "blocks"
            )
            or []
        )

        text_characters = len(
            "".join(
                text.split()
            )
        )

        word_count = len(
            words
        )

        image_count = len(
            images
        )

        drawing_count = len(
            drawings
        )

        block_count = len(
            blocks
        )

        has_text = (
            text_characters > 0
        )

        has_images = (
            image_count > 0
        )

        image_only = (
            has_images
            and
            not has_text
        )

        return PageIntegrity(
            page_number=page_number,

            text_characters=text_characters,

            word_count=word_count,

            image_count=image_count,

            drawing_count=drawing_count,

            block_count=block_count,

            has_text=has_text,

            has_images=has_images,

            image_only=image_only,
        )

    # ======================================================
    # PAGE RULES
    # ======================================================

    @staticmethod
    def _check_page(
        page: PageIntegrity,
    ) -> list[IntegrityFinding]:
        findings: list[
            IntegrityFinding
        ] = []

        if page.image_only:

            findings.append(
                IntegrityFinding(
                    rule_id=(
                        "PDF_IMAGE_ONLY_PAGE"
                    ),

                    category="page_structure",

                    severity=(
                        IntegritySeverity.LOW
                    ),

                    message=(
                        f"Page {page.page_number} "
                        "contains only images."
                    ),

                    reason=(
                        "The page has image content but no native "
                        "text layer. This can be normal for scanned "
                        "documents, but image-only pages make field "
                        "level verification and tamper analysis harder."
                    ),

                    evidence={
                        "page": (
                            page.page_number
                        ),

                        "image_count": (
                            page.image_count
                        ),

                        "text_characters": (
                            page.text_characters
                        ),
                    },

                    score=10.0,
                )
            )

        return findings

    # ======================================================
    # FONT ANALYSIS
    # ======================================================

    @staticmethod
    def _check_fonts(
        document: fitz.Document,
    ) -> list[IntegrityFinding]:
        """
        Detect unusual font diversity.

        Font diversity alone is not evidence of manipulation.

        The purpose is to identify pages worth deeper inspection.
        """

        findings: list[
            IntegrityFinding
        ] = []

        total_fonts = 0

        font_names: set[str] = set()

        try:

            for page_number in range(
                document.page_count
            ):

                page = document.load_page(
                    page_number
                )

                fonts = (
                    page.get_fonts(
                        full=True
                    )
                    or []
                )

                total_fonts += len(
                    fonts
                )

                for font in fonts:

                    if len(font) >= 4:

                        name = str(
                            font[3]
                            or ""
                        ).strip()

                        if name:
                            font_names.add(
                                name
                            )

        except Exception as exc:
            logger.debug(
                "Unable to inspect fonts: %s",
                exc,
            )

        # Very high font diversity can be a review signal.
        if (
            total_fonts >= 30
            and
            len(font_names) >= 20
        ):

            findings.append(
                IntegrityFinding(
                    rule_id=(
                        "PDF_HIGH_FONT_DIVERSITY"
                    ),

                    category="structure",

                    severity=(
                        IntegritySeverity.LOW
                    ),

                    message=(
                        "The PDF contains unusually high "
                        "font diversity."
                    ),

                    reason=(
                        "A large number of distinct fonts can occur "
                        "naturally in complex PDFs, but it can also "
                        "appear after document reconstruction or "
                        "editing. It is therefore treated only as "
                        "a weak review signal."
                    ),

                    evidence={
                        "font_objects": (
                            total_fonts
                        ),

                        "unique_fonts": (
                            len(font_names)
                        ),
                    },

                    score=10.0,
                )
            )

        return findings

    # ======================================================
    # STRUCTURE
    # ======================================================

    @staticmethod
    def _check_structure(
        document: fitz.Document,
    ) -> list[IntegrityFinding]:
        findings: list[
            IntegrityFinding
        ] = []

        try:
            xref_length = (
                document.xref_length()
            )

            if (
                document.page_count > 0
                and
                xref_length <= 5
            ):

                findings.append(
                    IntegrityFinding(
                        rule_id=(
                            "PDF_UNUSUAL_OBJECT_STRUCTURE"
                        ),

                        category="structure",

                        severity=(
                            IntegritySeverity.LOW
                        ),

                        message=(
                            "The PDF contains an unusually "
                            "small object structure."
                        ),

                        reason=(
                            "The number of PDF objects is unusually "
                            "small relative to the page count. This "
                            "can occur with simplified or reconstructed "
                            "documents and should be treated only as "
                            "a weak signal."
                        ),

                        evidence={
                            "xref_length": (
                                xref_length
                            ),

                            "page_count": (
                                document.page_count
                            ),
                        },

                        score=10.0,
                    )
                )

        except Exception as exc:
            logger.debug(
                "Unable to inspect PDF structure: %s",
                exc,
            )

        return findings

    # ======================================================
    # MIXED CONTENT
    # ======================================================

    @staticmethod
    def _check_mixed_content(
        pages: list[PageIntegrity],
    ) -> list[IntegrityFinding]:
        """
        Detect documents containing both native-text and
        image-only pages.

        This is not inherently suspicious.
        """

        if not pages:
            return []

        image_only_pages = [
            page.page_number
            for page in pages
            if page.image_only
        ]

        native_text_pages = [
            page.page_number
            for page in pages
            if page.has_text
        ]

        if (
            image_only_pages
            and
            native_text_pages
        ):

            return [
                IntegrityFinding(
                    rule_id=(
                        "PDF_MIXED_TEXT_IMAGE_STRUCTURE"
                    ),

                    category="page_structure",

                    severity=(
                        IntegritySeverity.LOW
                    ),

                    message=(
                        "The document mixes native-text "
                        "and image-only pages."
                    ),

                    reason=(
                        "Mixed page representations can be legitimate "
                        "in scanned or assembled documents. However, "
                        "different page representations make consistent "
                        "tamper analysis more difficult."
                    ),

                    evidence={
                        "image_only_pages": (
                            image_only_pages
                        ),

                        "native_text_pages": (
                            native_text_pages
                        ),
                    },

                    score=10.0,
                )
            ]

        return []

    # ======================================================
    # RISK CALCULATION
    # ======================================================

    @staticmethod
    def _calculate_risk(
        findings: list[
            IntegrityFinding
        ],
    ) -> float:
        """
        Calculate bounded numerical risk.

        Critical/high findings retain priority.
        """

        if not findings:
            return 0.0

        scores = sorted(
            (
                max(
                    0.0,
                    min(
                        100.0,
                        finding.score,
                    ),
                )

                for finding in findings
            ),
            reverse=True,
        )

        score = (
            scores[0]
            +
            sum(
                value * 0.35
                for value
                in scores[1:]
            )
        )

        return round(
            min(
                1.0,
                score / 100.0,
            ),
            4,
        )

    # ======================================================
    # RISK LEVEL
    # ======================================================

    @staticmethod
    def _risk_level(
        findings: list[
            IntegrityFinding
        ],

        risk_score: float,
    ) -> IntegritySeverity:
        """
        Explicit finding severity takes precedence over score.
        """

        severities = [
            finding.severity
            for finding in findings
        ]

        if (
            IntegritySeverity.CRITICAL
            in severities
        ):
            return IntegritySeverity.CRITICAL

        if (
            IntegritySeverity.HIGH
            in severities
        ):
            return IntegritySeverity.HIGH

        if (
            IntegritySeverity.MEDIUM
            in severities
        ):
            return IntegritySeverity.MEDIUM

        if (
            IntegritySeverity.LOW
            in severities
        ):
            return IntegritySeverity.LOW

        if risk_score >= 0.75:
            return IntegritySeverity.CRITICAL

        if risk_score >= 0.50:
            return IntegritySeverity.HIGH

        if risk_score >= 0.25:
            return IntegritySeverity.MEDIUM

        if risk_score > 0.0:
            return IntegritySeverity.LOW

        return IntegritySeverity.INFO

    # ======================================================
    # STATUS
    # ======================================================

    @staticmethod
    def _status(
        risk_level: IntegritySeverity,
    ) -> IntegrityStatus:

        if (
            risk_level
            == IntegritySeverity.CRITICAL
        ):
            return IntegrityStatus.HIGH_RISK

        if (
            risk_level
            == IntegritySeverity.HIGH
        ):
            return IntegrityStatus.HIGH_RISK

        if (
            risk_level
            == IntegritySeverity.MEDIUM
        ):
            return IntegrityStatus.MEDIUM_RISK

        if (
            risk_level
            == IntegritySeverity.LOW
        ):
            return IntegrityStatus.LOW_RISK

        return IntegrityStatus.CLEAN

    # ======================================================
    # REASON
    # ======================================================

    @staticmethod
    def _build_reason(
        findings: list[
            IntegrityFinding
        ],

        status: IntegrityStatus,
    ) -> str:

        if not findings:

            return (
                "No significant technical PDF integrity "
                "anomaly was detected."
            )

        ordered = sorted(
            findings,
            key=lambda finding: (
                finding.severity_rank,
                finding.score,
            ),
            reverse=True,
        )

        primary = ordered[0]

        if (
            status
            == IntegrityStatus.HIGH_RISK
        ):

            return (
                primary.reason
                or primary.message
            )

        if (
            status
            == IntegrityStatus.MEDIUM_RISK
        ):

            return (
                primary.reason
                or primary.message
            )

        return (
            "Minor PDF integrity signals were detected, "
            "but they are not sufficient by themselves "
            "to classify the document as fake."
        )

    # ======================================================
    # SUMMARY
    # ======================================================

    @staticmethod
    def _build_summary(
        findings: list[
            IntegrityFinding
        ],

        status: IntegrityStatus,
    ) -> str:

        if (
            status
            == IntegrityStatus.CLEAN
        ):
            return (
                "No significant PDF integrity anomalies detected."
            )

        if (
            status
            == IntegrityStatus.LOW_RISK
        ):
            return (
                "Minor PDF integrity signals detected."
            )

        if (
            status
            == IntegrityStatus.MEDIUM_RISK
        ):
            return (
                "Some PDF integrity anomalies require review."
            )

        return (
            "High-risk PDF integrity anomalies detected."
        )

    # ======================================================
    # CONFIDENCE
    # ======================================================

    @staticmethod
    def _confidence(
        findings: list[
            IntegrityFinding
        ],

        risk_level: IntegritySeverity,
    ) -> float:

        if (
            risk_level
            == IntegritySeverity.CRITICAL
        ):
            return 0.95

        if (
            risk_level
            == IntegritySeverity.HIGH
        ):
            return 0.90

        if (
            risk_level
            == IntegritySeverity.MEDIUM
        ):
            return 0.75

        if (
            risk_level
            == IntegritySeverity.LOW
        ):
            return 0.60

        return 0.55


# ==========================================================
# CONVENIENCE FUNCTION
# ==========================================================


def analyze_pdf_integrity(
    pdf_bytes: bytes,
) -> PDFIntegrityResult:
    """
    Convenience wrapper.
    """

    return PDFIntegrityAnalyzer().analyze(
        pdf_bytes
    )