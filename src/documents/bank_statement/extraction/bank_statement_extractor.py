"""
Generic Bank Statement Extractor.

Phase 2 - Document Intelligence / Extraction.

Responsibilities:
- orchestrate the complete Phase 2 extraction pipeline
- route files to native PDF extraction or OCR
- when OCR is required, run a tiered accuracy pipeline:
      Tier 1: PP-OCR            (fast, default)
      Tier 2: PP-StructureV3    (table/layout aware fallback)
      Tier 3: PaddleOCR-VL      (VLM-based fallback, last resort)
  and keep whichever tier produced the most confident result
- normalize extracted text
- identify statement structure
- extract generic statement metadata
- parse transactions
- return one standardized bank-independent result

Important:
This module does NOT:
- detect tampering
- calculate fraud/risk scores
- perform loan eligibility logic
- contain bank-specific extraction rules

Pipeline:

    uploaded bytes
        |
        v
    ExtractionRouter
        |
        +---- native_pdf ----> NativePDFExtractor
        |
        +---- ocr_pdf / ocr_image
                |
                v
              Tier 1: PP-OCR (ocr_extractor)
                |  low quality / no transactions?
                v
              Tier 2: PP-StructureV3 (paddle_structure_extractor)
                |  still low quality / no transactions?
                v
              Tier 3: PaddleOCR-VL (paddle_vl_extractor)
                |
                v
          best-scoring tier's pages
                |
                +--> LayoutReconstructor (tiers with token geometry)
                |     -> TransactionAssembler
                |     -> reconcile_structured()
                |
                +--> TransactionParser.parse() (text-only tiers,
                      e.g. PaddleOCR-VL markdown, and native PDFs)
        |
        v
    TextNormalizer -> StructureParser -> MetadataExtractor
        |
        v
    BankStatementExtractionResult
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.documents.bank_statement.extraction.extraction_router import (
    extraction_router,
)
from src.documents.bank_statement.extraction.metadata_extractor import (
    metadata_extractor,
)
from src.documents.bank_statement.extraction.native_pdf_extractor import (
    native_pdf_extractor,
)
from src.documents.bank_statement.extraction.ocr_extractor import (
    ocr_extractor,
)
from src.documents.bank_statement.extraction.paddle_structure_extractor import (
    paddle_structure_extractor,
)
from src.documents.bank_statement.extraction.paddle_vl_extractor import (
    paddle_vl_extractor,
)
from src.documents.bank_statement.extraction.layout_reconstructor import (
    layout_reconstructor,
)
from src.documents.bank_statement.extraction.transaction_assembler import (
    transaction_assembler,
)
from src.documents.bank_statement.extraction.structure_parser import (
    structure_parser,
)
from src.documents.bank_statement.extraction.text_normalizer import (
    text_normalizer,
)
from src.documents.bank_statement.extraction.transaction_parser import (
    transaction_parser,
)


# ============================================================
# Result model
# ============================================================


@dataclass(frozen=True)
class BankStatementExtractionResult:
    """
    Standardized Phase 2 extraction result.

    The result intentionally contains generic structures rather
    than bank-specific output models.
    """

    filename: str
    detected_type: str

    extraction_method: str
    page_count: int

    text_char_count: int
    line_count: int

    native_text_available: bool
    ocr_used: bool
    ocr_engine: str | None
    ocr_confidence: float | None

    transaction_header_detected: bool
    transaction_region_detected: bool

    metadata: dict[str, Any]

    opening_balance: Any

    transaction_count: int
    rejected_transaction_blocks: int
    unresolved_direction_count: int
    reconciled_transaction_count: int

    transaction_parser_confidence: float

    transactions: tuple[dict[str, Any], ...]

    normalized_text: str

    # ------------------------------------------------------
    # Tiered-pipeline reporting.
    #
    # Added without removing/renaming any existing field, so
    # existing consumers of this dataclass keep working.
    # ------------------------------------------------------

    # Which extraction tier ultimately produced this result:
    # "native_pdf" | "paddleocr" | "pp-structurev3" | "paddleocr-vl"
    extraction_tier: str = "native_pdf"

    # Overall confidence in the extracted result, combining OCR
    # confidence, transaction-parser confidence, and whether a
    # transaction table was actually resolved. Always in [0, 1].
    quality_score: float = 1.0

    # Compact summary block mirroring the shape used by the
    # Phase 1 detection endpoint's `text_extraction` object, for
    # a consistent API surface across phases.
    text_extraction: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Internal per-tier candidate
# ============================================================


@dataclass
class _TierCandidate:
    engine_label: str
    ocr_used: bool
    ocr_engine: str | None
    ocr_confidence: float | None
    page_count: int

    normalized_text: str
    text_char_count: int
    line_count: int

    transaction_header_detected: bool
    transaction_region_detected: bool

    metadata: dict[str, Any]
    opening_balance: Any

    transaction_count: int
    rejected_transaction_blocks: int
    unresolved_direction_count: int
    reconciled_transaction_count: int
    transaction_parser_confidence: float
    transactions: tuple[dict[str, Any], ...]

    quality_score: float
    error: str | None = None


# ============================================================
# Main extractor
# ============================================================


class BankStatementExtractor:
    """
    Generic Phase 2 bank-statement extraction orchestrator.

    Coordinates the modular Phase 2 extraction components.

    No bank-specific parsing logic belongs here.
    """

    # --------------------------------------------------------
    # Quality thresholds controlling tier fallback.
    #
    # These are intentionally conservative: falling back to a
    # slower tier only happens when the faster tier's result
    # looks unreliable, not merely "not perfect".
    # --------------------------------------------------------

    STRUCTURE_FALLBACK_THRESHOLD = 0.72
    VL_FALLBACK_THRESHOLD = 0.55

    # ========================================================
    # Public API
    # ========================================================

    def extract(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> BankStatementExtractionResult:
        """
        Run the complete Phase 2 extraction pipeline.

        Parameters
        ----------
        file_bytes:
            Original uploaded document bytes.

        filename:
            Original uploaded filename.

        Returns
        -------
        BankStatementExtractionResult
            Standardized extracted statement information.
        """

        route = extraction_router.route(
            file_bytes=file_bytes,
            filename=filename,
        )

        if route.extraction_method == "native_pdf":
            candidate = self._run_native_tier(
                file_bytes=file_bytes,
                filename=filename,
            )
            return self._finalize(
                filename=filename,
                detected_type=route.detected_type,
                extraction_method=route.extraction_method,
                native_text_available=route.native_text_available,
                candidate=candidate,
            )

        if route.extraction_method in {"ocr_pdf", "ocr_image"}:
            candidate = self._run_ocr_tiers(
                file_bytes=file_bytes,
                filename=filename,
                detected_type=route.detected_type,
            )
            return self._finalize(
                filename=filename,
                detected_type=route.detected_type,
                extraction_method=route.extraction_method,
                native_text_available=route.native_text_available,
                candidate=candidate,
            )

        raise ValueError(
            "Unsupported extraction method returned by extraction "
            f"router: {route.extraction_method}"
        )

    # ========================================================
    # NATIVE PDF TIER
    # ========================================================

    def _run_native_tier(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> _TierCandidate:

        result = native_pdf_extractor.extract(file_bytes, filename)

        return self._process_text_source(
            raw_text=result.text,
            page_count=result.page_count,
            ocr_used=False,
            ocr_engine=None,
            ocr_confidence=None,
            ocr_pages=None,
            engine_label="native_pdf",
            use_spatial=False,
        )

    # ========================================================
    # OCR TIER ORCHESTRATION
    # ========================================================

    def _run_ocr_tiers(
        self,
        file_bytes: bytes,
        filename: str,
        detected_type: str,
    ) -> _TierCandidate:
        """
        Run PP-OCR first, then escalate to PP-StructureV3 and
        finally PaddleOCR-VL only if quality remains low.

        Every tier that raises (e.g. an optional dependency is not
        installed, or the model fails on this page) is treated as
        a failed candidate rather than aborting the request - the
        best candidate found so far is still returned.
        """

        best: _TierCandidate | None = None

        # ---------------- Tier 1: PP-OCR ----------------------

        try:
            tier1_result = ocr_extractor.extract(
                file_bytes=file_bytes,
                filename=filename,
                detected_type=detected_type,
            )

            best = self._process_text_source(
                raw_text=tier1_result.text,
                page_count=tier1_result.page_count,
                ocr_used=True,
                ocr_engine=tier1_result.engine,
                ocr_confidence=tier1_result.average_confidence,
                ocr_pages=tier1_result.pages,
                engine_label="paddleocr",
                use_spatial=True,
            )
        except Exception as exc:
            best = self._failed_candidate("paddleocr", exc)

        if self._is_good_enough(best, self.STRUCTURE_FALLBACK_THRESHOLD):
            return best

        # ---------------- Tier 2: PP-StructureV3 ---------------

        try:
            tier2_result = paddle_structure_extractor.extract(
                file_bytes=file_bytes,
                filename=filename,
                detected_type=detected_type,
            )

            tier2_candidate = self._process_text_source(
                raw_text=tier2_result.text,
                page_count=tier2_result.page_count,
                ocr_used=True,
                ocr_engine=tier2_result.engine,
                ocr_confidence=tier2_result.average_confidence,
                ocr_pages=tier2_result.pages,
                engine_label="pp-structurev3",
                use_spatial=True,
            )

            best = self._better_candidate(best, tier2_candidate)
        except Exception as exc:
            failed = self._failed_candidate("pp-structurev3", exc)
            best = self._better_candidate(best, failed)

        if self._is_good_enough(best, self.VL_FALLBACK_THRESHOLD):
            return best

        # ---------------- Tier 3: PaddleOCR-VL ------------------

        try:
            tier3_result = paddle_vl_extractor.extract(
                file_bytes=file_bytes,
                filename=filename,
                detected_type=detected_type,
            )

            tier3_candidate = self._process_text_source(
                raw_text=tier3_result.text,
                page_count=tier3_result.page_count,
                ocr_used=True,
                ocr_engine=tier3_result.engine,
                ocr_confidence=tier3_result.average_confidence,
                ocr_pages=None,  # markdown text only, no geometry
                engine_label="paddleocr-vl",
                use_spatial=False,
            )

            best = self._better_candidate(best, tier3_candidate)
        except Exception as exc:
            failed = self._failed_candidate("paddleocr-vl", exc)
            best = self._better_candidate(best, failed)

        if best is None or best.error is not None:
            error_detail = best.error if best else "no extraction tier produced a result"
            raise ValueError(
                "Unable to extract a usable bank statement after "
                f"trying all OCR tiers: {error_detail}"
            )

        return best

    # ========================================================
    # SHARED TEXT -> STRUCTURE -> TRANSACTIONS PIPELINE
    # ========================================================

    def _process_text_source(
        self,
        raw_text: str,
        page_count: int,
        ocr_used: bool,
        ocr_engine: str | None,
        ocr_confidence: float | None,
        ocr_pages: tuple[Any, ...] | None,
        engine_label: str,
        use_spatial: bool,
    ) -> _TierCandidate:
        """
        Run steps common to every tier: normalize -> structure ->
        metadata -> transactions -> quality score.

        `use_spatial=True` requires `ocr_pages` with real token
        geometry (PP-OCR / PP-StructureV3). `use_spatial=False`
        parses the flattened text directly (native PDF text, or
        PaddleOCR-VL markdown).
        """

        if not raw_text or not raw_text.strip():
            raise ValueError(
                "No usable text could be extracted from the document."
            )

        normalized = text_normalizer.normalize(raw_text)

        if not normalized.text.strip():
            raise ValueError(
                "Extracted document text became empty after normalization."
            )

        structure = structure_parser.parse(normalized.text)

        if not structure.transaction_region_detected:
            raise ValueError(
                "Unable to identify a transaction region in the "
                "extracted document."
            )

        header_text = "\n".join(structure.header_lines).strip()
        body_text = "\n".join(structure.body_lines).strip()

        if not body_text:
            raise ValueError(
                "Transaction region was detected but contains no "
                "usable text."
            )

        metadata_result = metadata_extractor.extract(header_text)
        metadata = metadata_result.to_dict()

        transaction_header = self._resolve_transaction_header(
            normalized_text=normalized.text,
            header_lines=structure.header_lines,
            transaction_header_line=structure.transaction_header_line,
            transaction_header_detected=structure.transaction_header_detected,
        )

        if use_spatial and ocr_pages:
            layout_result = layout_reconstructor.reconstruct(ocr_pages)

            structured_rows = tuple(
                row
                for page in layout_result.pages
                for row in page.structured_rows
            )

            assembly_result = transaction_assembler.assemble(structured_rows)

            transaction_result = transaction_parser.reconcile_structured(
                assembly_result.transactions,
                header_text=header_text,
            )
        else:
            transaction_result = transaction_parser.parse(
                body_text,
                transaction_header,
                header_text=header_text,
            )

        transactions = tuple(
            transaction.to_dict() for transaction in transaction_result.transactions
        )

        quality_score = self._score_candidate(
            ocr_confidence=ocr_confidence,
            parser_confidence=transaction_result.parser_confidence,
            transaction_count=transaction_result.transaction_count,
            transaction_region_detected=structure.transaction_region_detected,
        )

        return _TierCandidate(
            engine_label=engine_label,
            ocr_used=ocr_used,
            ocr_engine=ocr_engine,
            ocr_confidence=ocr_confidence,
            page_count=page_count,
            normalized_text=normalized.text,
            text_char_count=normalized.char_count,
            line_count=normalized.line_count,
            transaction_header_detected=structure.transaction_header_detected,
            transaction_region_detected=structure.transaction_region_detected,
            metadata=metadata,
            opening_balance=transaction_result.opening_balance,
            transaction_count=transaction_result.transaction_count,
            rejected_transaction_blocks=transaction_result.rejected_blocks,
            unresolved_direction_count=transaction_result.unresolved_direction_count,
            reconciled_transaction_count=transaction_result.reconciled_count,
            transaction_parser_confidence=transaction_result.parser_confidence,
            transactions=transactions,
            quality_score=quality_score,
        )

    # ========================================================
    # QUALITY SCORING / TIER SELECTION
    # ========================================================

    @staticmethod
    def _score_candidate(
        ocr_confidence: float | None,
        parser_confidence: float,
        transaction_count: int,
        transaction_region_detected: bool,
    ) -> float:
        """
        Combine OCR confidence, transaction-parser confidence, and
        whether any transactions were actually resolved into one
        [0, 1] quality score used to decide whether to escalate to
        the next extraction tier.

        Native PDFs have no OCR confidence, so that term is
        treated as fully confident (1.0) - native text extraction
        does not suffer from recognition error.
        """

        if not transaction_region_detected:
            return 0.0

        recognition_term = ocr_confidence if ocr_confidence is not None else 1.0
        recognition_term = max(0.0, min(1.0, recognition_term))

        parser_term = max(0.0, min(1.0, parser_confidence))

        yield_term = 1.0 if transaction_count > 0 else 0.0

        score = (
            0.3 * recognition_term
            + 0.4 * parser_term
            + 0.3 * yield_term
        )

        return max(0.0, min(1.0, score))

    @classmethod
    def _is_good_enough(
        cls,
        candidate: _TierCandidate | None,
        threshold: float,
    ) -> bool:

        if candidate is None or candidate.error is not None:
            return False

        return (
            candidate.transaction_count > 0
            and candidate.quality_score >= threshold
        )

    @staticmethod
    def _better_candidate(
        current: _TierCandidate | None,
        challenger: _TierCandidate | None,
    ) -> _TierCandidate | None:

        if challenger is None or challenger.error is not None:
            return current

        if current is None or current.error is not None:
            return challenger

        # Prefer more resolved transactions first, then higher
        # overall quality score.
        if challenger.transaction_count != current.transaction_count:
            return (
                challenger
                if challenger.transaction_count > current.transaction_count
                else current
            )

        return challenger if challenger.quality_score > current.quality_score else current

    @staticmethod
    def _failed_candidate(engine_label: str, exc: Exception) -> _TierCandidate:
        return _TierCandidate(
            engine_label=engine_label,
            ocr_used=True,
            ocr_engine=engine_label,
            ocr_confidence=None,
            page_count=0,
            normalized_text="",
            text_char_count=0,
            line_count=0,
            transaction_header_detected=False,
            transaction_region_detected=False,
            metadata={},
            opening_balance=None,
            transaction_count=0,
            rejected_transaction_blocks=0,
            unresolved_direction_count=0,
            reconciled_transaction_count=0,
            transaction_parser_confidence=0.0,
            transactions=(),
            quality_score=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )

    # ========================================================
    # FINAL RESULT ASSEMBLY
    # ========================================================

    @staticmethod
    def _finalize(
        filename: str,
        detected_type: str,
        extraction_method: str,
        native_text_available: bool,
        candidate: _TierCandidate,
    ) -> BankStatementExtractionResult:

        if candidate.error is not None:
            raise ValueError(candidate.error)

        text_extraction = {
            "method": candidate.engine_label,
            "ocr_used": candidate.ocr_used,
            "ocr_engine": candidate.ocr_engine,
            "quality_score": round(candidate.quality_score, 3),
        }

        return BankStatementExtractionResult(
            filename=filename,
            detected_type=detected_type,
            extraction_method=extraction_method,
            page_count=candidate.page_count,
            text_char_count=candidate.text_char_count,
            line_count=candidate.line_count,
            native_text_available=native_text_available,
            ocr_used=candidate.ocr_used,
            ocr_engine=candidate.ocr_engine,
            ocr_confidence=candidate.ocr_confidence,
            transaction_header_detected=candidate.transaction_header_detected,
            transaction_region_detected=candidate.transaction_region_detected,
            metadata=candidate.metadata,
            opening_balance=candidate.opening_balance,
            transaction_count=candidate.transaction_count,
            rejected_transaction_blocks=candidate.rejected_transaction_blocks,
            unresolved_direction_count=candidate.unresolved_direction_count,
            reconciled_transaction_count=candidate.reconciled_transaction_count,
            transaction_parser_confidence=candidate.transaction_parser_confidence,
            transactions=candidate.transactions,
            normalized_text=candidate.normalized_text,
            extraction_tier=candidate.engine_label,
            quality_score=candidate.quality_score,
            text_extraction=text_extraction,
        )

    # ========================================================
    # Transaction header resolution
    # ========================================================

    @staticmethod
    def _resolve_transaction_header(
        normalized_text: str,
        header_lines: list[str] | tuple[str, ...],
        transaction_header_line: int | None,
        transaction_header_detected: bool,
    ) -> str:
        """
        Resolve the actual transaction-column header text.

        StructureParser exposes transaction_header_line as the
        zero-based line position in the normalized document.

        TransactionParser, however, expects the actual textual
        header because it uses the header to infer column
        semantics such as:

            debit / credit / withdrawal / deposit / balance

        This method bridges those two interfaces without
        introducing bank-specific logic.
        """

        if not transaction_header_detected:
            return ""

        document_lines = normalized_text.splitlines()

        # ----------------------------------------------------
        # Primary strategy:
        # use StructureParser's detected line index.
        # ----------------------------------------------------

        if isinstance(transaction_header_line, int):
            if 0 <= transaction_header_line < len(document_lines):
                candidate = document_lines[transaction_header_line].strip()
                if candidate:
                    return candidate

        # ----------------------------------------------------
        # Defensive fallback:
        # search the header region for a line that looks like
        # a transaction-column header.
        #
        # Generic concepts only. No bank names or exact
        # bank-specific header formats are used.
        # ----------------------------------------------------

        for line in reversed(list(header_lines)):
            candidate = line.strip()

            if not candidate:
                continue

            lowered = candidate.lower()

            has_date = "date" in lowered
            has_balance = "balance" in lowered

            has_transaction_field = any(
                token in lowered
                for token in (
                    "description",
                    "particular",
                    "narration",
                    "remarks",
                    "withdrawal",
                    "deposit",
                    "debit",
                    "credit",
                )
            )

            if has_date and has_balance and has_transaction_field:
                return candidate

        # ----------------------------------------------------
        # Last defensive fallback.
        #
        # Returning an empty string is preferable to returning
        # an integer or unrelated line. TransactionParser can
        # then use its other inference/reconciliation logic.
        # ----------------------------------------------------

        return ""


# ============================================================
# Module-level service instance
# ============================================================


bank_statement_extractor = BankStatementExtractor()