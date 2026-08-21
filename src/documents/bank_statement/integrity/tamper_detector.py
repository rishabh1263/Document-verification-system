"""
Generic PDF Tamper Detector.

Combines independent PDF integrity evidence using evidence tiers.

Evidence sources:

1. PDFIntegrityAnalyzer
2. PageConsistencyAnalyzer
3. ContentStreamAnalyzer

Design principle:

Weak anomalies should create observations.

Strong or correlated independent anomalies should increase
tamper suspicion.

This prevents legitimate layout variation from automatically
being classified as document tampering.

Important:
- No OCR.
- No bank-specific templates.
- No transaction extraction.
- No hardcoded page numbers.
- Result is risk evidence, not forensic proof.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .content_stream_analyzer import (
    ContentStreamAnalysisResult,
)
from .page_consistency_analyzer import (
    PageConsistencyResult,
)
from .pdf_integrity_analyzer import (
    PDFIntegrityResult,
)


# ============================================================
# Result
# ============================================================


@dataclass(frozen=True)
class TamperDetectionResult:

    tamper_suspected: bool

    risk_score: int
    risk_level: str

    signals: tuple[str, ...]
    warnings: tuple[str, ...]

    integrity_checks_used: bool
    consistency_checks_used: bool
    content_stream_checks_used: bool

    structural_outlier_count: int
    structural_outlier_pages: tuple[int, ...]

    local_anomaly_count: int
    local_anomaly_pages: tuple[int, ...]

    strong_evidence_count: int
    moderate_evidence_count: int
    weak_evidence_count: int

    correlated_evidence_pages: tuple[int, ...]

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Detector
# ============================================================


class TamperDetector:

    """
    Evidence fusion for PDF anomaly detection.

    Score philosophy:

        Weak evidence       → small/no risk increase
        Moderate evidence   → limited increase
        Strong evidence     → significant increase
        Correlated evidence → significant increase

    A typography anomaly alone must not produce a tamper
    verdict.
    """

    # ========================================================
    # Public API
    # ========================================================

    def detect(
        self,
        integrity: PDFIntegrityResult,
        consistency: PageConsistencyResult | None = None,
        content_stream: ContentStreamAnalysisResult | None = None,
    ) -> TamperDetectionResult:

        score = 0

        signals: list[str] = []
        warnings: list[str] = []

        weak_evidence = 0
        moderate_evidence = 0
        strong_evidence = 0

        structural_pages: tuple[int, ...] = ()
        local_pages: tuple[int, ...] = ()
        correlated_pages: tuple[int, ...] = ()

        local_anomaly_count = 0

        # ====================================================
        # 1. BASIC PDF INTEGRITY
        # ====================================================

        if integrity.needs_password:

            weak_evidence += 1

            warnings.append(
                "PDF requires a password; some integrity "
                "checks may be unavailable."
            )

        # ----------------------------------------------------
        # Missing metadata is NOT tamper evidence.
        # ----------------------------------------------------

        if not integrity.metadata_available:

            warnings.append(
                "Descriptive PDF metadata is absent."
            )

        # ----------------------------------------------------
        # Annotations
        # ----------------------------------------------------

        if integrity.annotation_count > 0:

            weak_evidence += 1

            signals.append(
                f"{integrity.annotation_count} PDF "
                f"annotation(s) detected."
            )

        # ----------------------------------------------------
        # Embedded files
        # ----------------------------------------------------

        if integrity.embedded_file_count > 0:

            weak_evidence += 1

            signals.append(
                f"{integrity.embedded_file_count} embedded "
                f"file(s) detected."
            )

        # ----------------------------------------------------
        # Mixed text / image pages
        # ----------------------------------------------------

        if (
            integrity.pages_without_text > 0
            and integrity.pages_with_text > 0
        ):

            weak_evidence += 1

            signals.append(
                "Mixed text-bearing and text-free PDF pages "
                "detected."
            )

        # ====================================================
        # 2. PAGE-LEVEL STRUCTURAL CONSISTENCY
        # ====================================================

        if consistency is not None:

            structural_pages = tuple(
                consistency
                .structural_outlier_pages
            )

            signals.append(
                "Page structural consistency score: "
                f"{consistency.consistency_score:.2f}%."
            )

            # ------------------------------------------------
            # Font differences are observations.
            # ------------------------------------------------

            if (
                consistency
                .font_outlier_pages
            ):

                weak_evidence += 1

                signals.append(
                    "Page-level font variation detected on "
                    "page(s): "
                    + self._page_list(
                        consistency
                        .font_outlier_pages
                    )
                    + "."
                )

            # ------------------------------------------------
            # Font-size differences are observations.
            # ------------------------------------------------

            if (
                consistency
                .font_size_outlier_pages
            ):

                weak_evidence += 1

                signals.append(
                    "Page-level typography variation "
                    "detected on page(s): "
                    + self._page_list(
                        consistency
                        .font_size_outlier_pages
                    )
                    + "."
                )

            # ------------------------------------------------
            # Text block differences
            # ------------------------------------------------

            if (
                consistency
                .text_block_outlier_pages
            ):

                weak_evidence += 1

                signals.append(
                    "Text-block structure differs on "
                    "page(s): "
                    + self._page_list(
                        consistency
                        .text_block_outlier_pages
                    )
                    + "."
                )

            # ------------------------------------------------
            # Drawing differences
            # ------------------------------------------------

            if (
                consistency
                .drawing_outlier_pages
            ):

                weak_evidence += 1

                signals.append(
                    "Drawing-structure variation detected "
                    "on page(s): "
                    + self._page_list(
                        consistency
                        .drawing_outlier_pages
                    )
                    + "."
                )

            # ------------------------------------------------
            # Combined structural outliers
            #
            # One structural outlier in a long statement is
            # weak evidence, not automatic tampering.
            # ------------------------------------------------

            structural_count = len(
                structural_pages
            )

            if structural_count > 0:

                ratio = (
                    structural_count
                    / max(
                        consistency.pages_analyzed,
                        1,
                    )
                )

                if (
                    structural_count >= 3
                    and ratio >= 0.10
                ):

                    moderate_evidence += 1

                else:

                    weak_evidence += 1

                signals.append(
                    f"{structural_count} combined "
                    f"structural outlier page(s) detected: "
                    f"{self._page_list(structural_pages)}."
                )

                warnings.append(
                    "Structural variation can be legitimate "
                    "for cover, summary, footer, logo, or "
                    "terms pages."
                )

        # ====================================================
        # 3. LOCAL PDF CONTENT ANALYSIS
        # ====================================================

        if content_stream is not None:

            local_pages = tuple(
                content_stream
                .suspicious_pages
            )

            local_anomaly_count = (
                content_stream
                .total_local_anomalies
            )

            signals.append(
                "Local PDF consistency score: "
                f"{content_stream.local_consistency_score:.2f}%."
            )

            # ------------------------------------------------
            # Weak local evidence
            # ------------------------------------------------

            if (
                content_stream
                .stream_length_outlier_pages
            ):

                weak_evidence += 1

                signals.append(
                    "Unusual content-stream size detected "
                    "on page(s): "
                    + self._page_list(
                        content_stream
                        .stream_length_outlier_pages
                    )
                    + "."
                )

            if (
                content_stream
                .isolated_font_pages
            ):

                signals.append(
                    "Locally isolated font usage detected "
                    "on page(s): "
                    + self._page_list(
                        content_stream
                        .isolated_font_pages
                    )
                    + "."
                )

            if (
                content_stream
                .isolated_font_size_pages
            ):

                signals.append(
                    "Locally isolated typography detected "
                    "on page(s): "
                    + self._page_list(
                        content_stream
                        .isolated_font_size_pages
                    )
                    + "."
                )

            if (
                content_stream
                .text_span_outlier_pages
            ):

                signals.append(
                    "Local text-span variation detected on "
                    "page(s): "
                    + self._page_list(
                        content_stream
                        .text_span_outlier_pages
                    )
                    + "."
                )

            # ------------------------------------------------
            # Moderate local evidence
            # ------------------------------------------------

            if (
                content_stream
                .stream_count_outlier_pages
            ):

                # Stream-count variation is common in legitimate PDFs.
                # Treat it as weak evidence unless independently
                # corroborated by stronger signals.
                weak_evidence += 1

                signals.append(
                    "Unusual content-stream count detected "
                    "on page(s): "
                    + self._page_list(
                        content_stream
                        .stream_count_outlier_pages
                    )
                    + "."
                )

            if (
                content_stream
                .xobject_outlier_pages
            ):

                # XObject variation is not sufficient by itself to
                # classify a genuine statement as tampered.
                weak_evidence += 1

                signals.append(
                    "Unusual XObject usage detected on "
                    "page(s): "
                    + self._page_list(
                        content_stream
                        .xobject_outlier_pages
                    )
                    + "."
                )

            # ------------------------------------------------
            # Strong local evidence
            # ------------------------------------------------

            if (
                content_stream
                .overlapping_text_pages
            ):

                strong_evidence += 1

                signals.append(
                    "Overlapping text objects detected on "
                    "page(s): "
                    + self._page_list(
                        content_stream
                        .overlapping_text_pages
                    )
                    + "."
                )

            if (
                content_stream
                .duplicate_overlay_pages
            ):

                strong_evidence += 2

                signals.append(
                    "Duplicate text-overlay indicators "
                    "detected on page(s): "
                    + self._page_list(
                        content_stream
                        .duplicate_overlay_pages
                    )
                    + "."
                )

            # ------------------------------------------------
            # Suspicious local pages
            # ------------------------------------------------

            if local_pages:

                signals.append(
                    f"{len(local_pages)} page(s) contain "
                    f"combined local PDF anomalies: "
                    f"{self._page_list(local_pages)}."
                )

        # ====================================================
        # 4. CROSS-DETECTOR CORRELATION
        # ====================================================

        if (
            consistency is not None
            and content_stream is not None
        ):

            correlated_pages = tuple(
                sorted(
                    set(
                        structural_pages
                    )
                    & set(
                        local_pages
                    )
                )
            )

            if correlated_pages:

                # Correlation is useful, but only when the
                # local analyzer itself classified the page
                # as suspicious using moderate/strong local
                # evidence.

                moderate_evidence += 1

                signals.append(
                    "Independent structural and local PDF "
                    "anomalies correlate on page(s): "
                    + self._page_list(
                        correlated_pages
                    )
                    + "."
                )

        # ====================================================
        # 5. EVIDENCE-TIER SCORING
        # ====================================================

        # Weak evidence intentionally has a small cap.
        #
        # Ten weak layout observations must not turn a normal
        # PDF into a HIGH-risk document.

        weak_score = min(
            weak_evidence * 3,
            15,
        )

        moderate_score = min(
            moderate_evidence * 10,
            30,
        )

        strong_score = min(
            strong_evidence * 30,
            60,
        )

        score = (
            weak_score
            + moderate_score
            + strong_score
        )

        score = min(
            score,
            100,
        )

        # ====================================================
        # 6. RISK LEVEL
        # ====================================================
        #
        # IMPORTANT:
        # Moderate observations are not automatically tampering.
        #
        # Genuine bank statements commonly contain:
        #   - different XObjects
        #   - different stream counts
        #   - footer/header changes
        #   - summary/transaction page differences
        #
        # Therefore MEDIUM requires corroborated evidence.

        if strong_evidence >= 2:
            risk_level = "HIGH"

        elif (
            strong_evidence >= 1
            or (
                moderate_evidence >= 3
                and score >= 30
            )
        ):
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

        # ====================================================
        # 7. TAMPER DECISION
        # ====================================================
        #
        # A genuine PDF must not be rejected merely because it has
        # one or two moderate structural observations.
        #
        # Actual tamper suspicion requires:
        #   - strong evidence, OR
        #   - multiple independent moderate signals, OR
        #   - correlated structural + local evidence.

        tamper_suspected = bool(
            strong_evidence >= 1
            or (
                moderate_evidence >= 3
                and score >= 30
            )
            or (
                correlated_pages
                and moderate_evidence >= 2
                and score >= 30
            )
        )

        # ====================================================
        # 8. HARD FORENSIC ESCALATION
        # ====================================================

        if content_stream is not None:
            if (
                content_stream.duplicate_overlay_pages
                or content_stream.overlapping_text_pages
            ):
                tamper_suspected = True
                risk_level = "HIGH"
                score = max(score, 60)

        # ====================================================
        # 9. CLEAN SIGNAL
        # ====================================================

        if (
            weak_evidence == 0
            and moderate_evidence == 0
            and strong_evidence == 0
        ):

            signals.append(
                "No material PDF integrity anomalies "
                "detected."
            )

        # ====================================================
        # Result
        # ====================================================

        return TamperDetectionResult(

            tamper_suspected=(
                tamper_suspected
            ),

            risk_score=(
                score
            ),

            risk_level=(
                risk_level
            ),

            signals=tuple(
                signals
            ),

            warnings=tuple(
                warnings
            ),

            integrity_checks_used=True,

            consistency_checks_used=(
                consistency is not None
            ),

            content_stream_checks_used=(
                content_stream is not None
            ),

            structural_outlier_count=len(
                structural_pages
            ),

            structural_outlier_pages=(
                structural_pages
            ),

            local_anomaly_count=(
                local_anomaly_count
            ),

            local_anomaly_pages=(
                local_pages
            ),

            strong_evidence_count=(
                strong_evidence
            ),

            moderate_evidence_count=(
                moderate_evidence
            ),

            weak_evidence_count=(
                weak_evidence
            ),

            correlated_evidence_pages=(
                correlated_pages
            ),
        )

    # ========================================================
    # Page Formatter
    # ========================================================

    @staticmethod
    def _page_list(
        pages: tuple[int, ...],
        limit: int = 15,
    ) -> str:

        if not pages:

            return ""

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


tamper_detector = TamperDetector()