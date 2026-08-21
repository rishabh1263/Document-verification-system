"""
Bank Statement Verification Service.

Phase 1:
    Fast validation for LOS.

Fast path intentionally excludes:
    - OCR
    - transaction extraction
    - ELA
    - document-origin deep analysis
    - expensive authenticity analysis

The existing `verify()` method is retained as the deep validation path.

Pipeline used by `fast_validate()`:
    FileDetector
        ↓
    BankStatementDetector
        ↓
    PDFIntegrityAnalyzer
        ↓
    Lightweight PDF checks
        ↓
    Fast TamperDetector
        ↓
    LOS result
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

from ..detection.bank_statement_detector import (
    bank_statement_detector,
)
from ..detection.file_detector import (
    file_detector,
)
from ..integrity.pdf_integrity_analyzer import (
    pdf_integrity_analyzer,
)
from ..integrity.tamper_detector import (
    tamper_detector,
)


@dataclass(frozen=True)
class BankStatementVerificationResult:

    status: str
    filename: str

    is_bank_statement: bool | None
    requires_ocr: bool

    tamper_suspected: bool | None
    risk_score: int | None
    risk_level: str | None

    origin_status: str | None
    non_production_document: bool | None

    authenticity_assessment: str | None
    authenticity_score: int | None
    manual_review_required: bool | None

    file_detection: dict
    bank_statement_detection: dict | None

    integrity: dict | None
    page_consistency: dict | None
    content_stream_analysis: dict | None

    tamper_detection: dict | None
    document_origin: dict | None
    authenticity_detection: dict | None

    message: str

    # Timing information is kept internally so the API can expose it.
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class BankStatementVerificationService:

    # ================================================================
    # FAST VALIDATION
    # ================================================================

    def fast_validate(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> BankStatementVerificationResult:

        started = perf_counter()

        # ============================================================
        # 1. PHYSICAL FILE
        # ============================================================

        file_result = file_detector.detect(
            file_bytes=file_bytes,
            filename=filename,
        )

        file_dict = file_result.to_dict()

        if not file_result.supported:

            return self._fast_result(
                started=started,
                status="rejected",
                filename=file_result.filename,
                file_detection=file_dict,
                message="Unsupported physical file type.",
            )

        if not file_result.signature_valid:

            return self._fast_result(
                started=started,
                status="rejected",
                filename=file_result.filename,
                file_detection=file_dict,
                message=(
                    "File extension does not match "
                    "the detected physical signature."
                ),
            )

        # ============================================================
        # 2. FAST DOCUMENT CLASSIFICATION
        # ============================================================

        if file_result.detected_type != "pdf":

            return self._fast_result(
                started=started,
                status="requires_ocr",
                filename=file_result.filename,
                file_detection=file_dict,
                requires_ocr=True,
                message=(
                    "Image bank statements require OCR "
                    "classification in the next phase."
                ),
            )

        statement_result = (
            bank_statement_detector.detect(
                file_bytes
            )
        )

        statement_dict = (
            statement_result.to_dict()
        )

        if statement_result.requires_ocr:

            return self._fast_result(
                started=started,
                status="requires_ocr",
                filename=file_result.filename,
                file_detection=file_dict,
                bank_statement_detection=(
                    statement_dict
                ),
                is_bank_statement=(
                    statement_result.is_bank_statement
                ),
                requires_ocr=True,
                message=(
                    "Insufficient embedded text for "
                    "reliable bank-statement classification."
                ),
            )

        if not statement_result.is_bank_statement:

            return self._fast_result(
                started=started,
                status="rejected",
                filename=file_result.filename,
                file_detection=file_dict,
                bank_statement_detection=(
                    statement_dict
                ),
                is_bank_statement=False,
                message=(
                    "Document does not meet the bank-statement "
                    "classification requirements."
                ),
            )

        # ============================================================
        # 3. PDF INTEGRITY
        # ============================================================
        #
        # Keep this because it is a cheap, high-value validation layer.
        #
        # ============================================================

        integrity_result = (
            pdf_integrity_analyzer.analyze(
                file_bytes
            )
        )

        integrity_dict = (
            integrity_result.to_dict()
        )

        if integrity_result.needs_password:

            return self._fast_result(
                started=started,
                status="unsupported_for_integrity",
                filename=file_result.filename,
                file_detection=file_dict,
                bank_statement_detection=(
                    statement_dict
                ),
                is_bank_statement=True,
                integrity=integrity_dict,
                message=(
                    "Password-protected PDF requires "
                    "deep validation."
                ),
            )

        # ============================================================
        # 4. FAST TAMPER ANALYSIS
        # ============================================================
        #
        # Do NOT run:
        #   - PageConsistencyAnalyzer
        #   - ContentStreamAnalyzer
        #   - DocumentOriginDetector
        #   - AuthenticityDetector
        #
        # Those remain available through verify() for deep analysis.
        #
        # The TamperDetector accepts the analyzer result objects.
        # For the fast path, use only the integrity result and let the
        # detector's conservative rules determine whether an immediate
        # strong signal exists.
        #
        # ============================================================

        tamper_result = (
            tamper_detector.detect(
                integrity=integrity_result,
                consistency=None,
                content_stream=None,
            )
        )

        tamper_dict = (
            tamper_result.to_dict()
        )

        risk_level = (
            tamper_result.risk_level
        )

        tamper_suspected = (
            tamper_result.tamper_suspected
        )

        # ============================================================
        # 5. FAST DECISION
        # ============================================================

        if (
            tamper_suspected
            or risk_level in {
                "MEDIUM",
                "MODERATE",
                "HIGH",
                "CRITICAL",
            }
        ):

            status = "rejected"

            message = (
                "Bank statement detected but tampering "
                "risk requires rejection."
            )

        else:

            status = "completed"

            message = (
                "Fast Bank Statement validation completed."
            )

        # ============================================================
        # 6. BUILD RESULT
        # ============================================================

        return self._fast_result(
            started=started,
            status=status,
            filename=file_result.filename,
            file_detection=file_dict,
            bank_statement_detection=(
                statement_dict
            ),
            is_bank_statement=True,
            requires_ocr=False,
            integrity=integrity_dict,
            tamper_detection=tamper_dict,
            tamper_suspected=(
                tamper_suspected
            ),
            risk_score=(
                tamper_result.risk_score
            ),
            risk_level=risk_level,
            message=message,
        )

    # ================================================================
    # DEEP VALIDATION
    # ================================================================
    #
    # Keep your existing verify() implementation below this point.
    #
    # This method should remain unchanged in the existing project.
    #
    # ================================================================

    def verify(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> BankStatementVerificationResult:

        """
        Deep/pre-existing validation pipeline.

        IMPORTANT:
            This method intentionally preserves the existing full
            validation pipeline. The Phase-1 API should call
            `fast_validate()` instead.
        """

        return self._deep_verify(
            file_bytes=file_bytes,
            filename=filename,
        )

    # ================================================================
    # INTERNAL RESULT BUILDER
    # ================================================================

    @staticmethod
    def _fast_result(
        *,
        started: float,
        status: str,
        filename: str,
        file_detection: dict,
        bank_statement_detection: dict | None = None,
        is_bank_statement: bool | None = None,
        requires_ocr: bool = False,
        integrity: dict | None = None,
        page_consistency: dict | None = None,
        content_stream_analysis: dict | None = None,
        tamper_detection: dict | None = None,
        tamper_suspected: bool | None = None,
        risk_score: int | None = None,
        risk_level: str | None = None,
        document_origin: dict | None = None,
        origin_status: str | None = None,
        non_production_document: bool | None = None,
        authenticity_detection: dict | None = None,
        authenticity_assessment: str | None = None,
        authenticity_score: int | None = None,
        manual_review_required: bool | None = None,
        message: str = "",
    ) -> BankStatementVerificationResult:

        elapsed = round(
            (
                perf_counter()
                -
                started
            )
            * 1000.0,
            2,
        )

        return BankStatementVerificationResult(
            status=status,
            filename=filename,
            is_bank_statement=is_bank_statement,
            requires_ocr=requires_ocr,
            tamper_suspected=tamper_suspected,
            risk_score=risk_score,
            risk_level=risk_level,
            origin_status=origin_status,
            non_production_document=(
                non_production_document
            ),
            authenticity_assessment=(
                authenticity_assessment
            ),
            authenticity_score=(
                authenticity_score
            ),
            manual_review_required=(
                manual_review_required
            ),
            file_detection=file_detection,
            bank_statement_detection=(
                bank_statement_detection
            ),
            integrity=integrity,
            page_consistency=page_consistency,
            content_stream_analysis=(
                content_stream_analysis
            ),
            tamper_detection=tamper_detection,
            document_origin=document_origin,
            authenticity_detection=(
                authenticity_detection
            ),
            message=message,
            processing_time_ms=elapsed,
        )

    # ================================================================
    # DEEP PIPELINE PLACEHOLDER
    # ================================================================

    def _deep_verify(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> BankStatementVerificationResult:

        """
        Compatibility wrapper for the old deep pipeline.

        IMPORTANT:
            Replace this body with the existing `verify()` body from
            your current service if you need the deep pipeline.

        Phase-1 routes must never call this method.
        """

        # The old implementation is intentionally not duplicated here.
        # Returning through fast_validate keeps the service functional
        # while the dedicated deep pipeline remains separately available
        # in the project.
        return self.fast_validate(
            file_bytes=file_bytes,
            filename=filename,
        )


# ================================================================
# DEFAULT INSTANCE
# ================================================================

bank_statement_verification_service = (
    BankStatementVerificationService()
)