"""
Integrity Signal Adapter
========================

Phase 4 adapter for Phase 1 integrity evidence.

Purpose
-------
Translate existing Phase 1 integrity findings into standardized
Phase 4 RiskSignal objects.

This adapter does NOT perform PDF forensic analysis itself.

Phase 1 remains responsible for:
- structural PDF checks
- extension / MIME consistency
- object-profile analysis
- suspicious PDF characteristics
- integrity findings

Phase 4 only consumes those findings.

Design principle
----------------
Preserve evidence without exaggerating it.

Unknown Phase 1 findings are retained as informational signals
rather than automatically treated as fraud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.documents.bank_statement.risk.models import (
    RiskSignal,
)


# ============================================================
# RESULT MODEL
# ============================================================


@dataclass(frozen=True)
class IntegritySignalSummary:
    """
    Standard Phase 4 integrity evidence summary.
    """

    finding_count: int

    risk_finding_count: int

    informational_count: int

    high_risk_count: int

    critical_count: int

    signal_count: int

    signals: tuple[
        RiskSignal,
        ...,
    ]

    confidence: float

    def to_dict(self) -> dict[str, Any]:

        return {
            "finding_count":
                self.finding_count,

            "risk_finding_count":
                self.risk_finding_count,

            "informational_count":
                self.informational_count,

            "high_risk_count":
                self.high_risk_count,

            "critical_count":
                self.critical_count,

            "signal_count":
                self.signal_count,

            "confidence":
                round(
                    self.confidence,
                    4,
                ),

            "signals": [
                signal.to_dict()
                for signal
                in self.signals
            ],
        }


# ============================================================
# ADAPTER
# ============================================================


class IntegritySignalAdapter:
    """
    Convert Phase 1 integrity findings to Phase 4 signals.
    """

    SEVERITY_SCORE = {
        "info": 0.0,
        "low": 2.0,
        "medium": 6.0,
        "high": 12.0,
        "critical": 20.0,
    }

    # ========================================================
    # PUBLIC API
    # ========================================================

    def analyze(
        self,
        integrity_result: Any,
    ) -> IntegritySignalSummary:
        """
        Convert Phase 1 integrity output into Phase 4 signals.

        The adapter supports both:
        - object-style Phase 1 results
        - dictionary-style Phase 1 results

        It looks for common finding containers such as:
            findings
            issues
            signals
            anomalies
            reasons
    """

        if integrity_result is None:

            return IntegritySignalSummary(
                finding_count=0,
                risk_finding_count=0,
                informational_count=0,
                high_risk_count=0,
                critical_count=0,
                signal_count=0,
                signals=(),
                confidence=0.0,
            )

        findings = self._extract_findings(
            integrity_result
        )

        signals: list[
            RiskSignal
        ] = []

        for finding in findings:

            signal = self._finding_to_signal(
                finding
            )

            if signal is not None:
                signals.append(
                    signal
                )

        # ----------------------------------------------------
        # Explicit top-level integrity flags
        # ----------------------------------------------------
        #
        # These are fallbacks for Phase 1 outputs that expose
        # summary flags but no detailed finding collection.
        # ----------------------------------------------------

        if not signals:

            fallback_signals = (
                self._top_level_fallback_signals(
                    integrity_result
                )
            )

            signals.extend(
                fallback_signals
            )

        risk_finding_count = sum(
            1
            for signal in signals
            if signal.score > 0
        )

        informational_count = sum(
            1
            for signal in signals
            if signal.score <= 0
        )

        high_risk_count = sum(
            1
            for signal in signals
            if signal.severity == "high"
        )

        critical_count = sum(
            1
            for signal in signals
            if signal.severity == "critical"
        )

        confidence = self._extract_confidence(
            integrity_result
        )

        if signals and confidence <= 0:
            # We have concrete evidence but Phase 1 did not
            # expose an aggregate confidence value.
            confidence = self._average_signal_confidence(
                signals
            )

        return IntegritySignalSummary(
            finding_count=len(
                findings
            )
            if findings
            else len(
                signals
            ),

            risk_finding_count=(
                risk_finding_count
            ),

            informational_count=(
                informational_count
            ),

            high_risk_count=(
                high_risk_count
            ),

            critical_count=(
                critical_count
            ),

            signal_count=len(
                signals
            ),

            signals=tuple(
                signals
            ),

            confidence=round(
                confidence,
                4,
            ),
        )

    # ========================================================
    # FINDING EXTRACTION
    # ========================================================

    def _extract_findings(
        self,
        integrity_result: Any,
    ) -> list[Any]:
        """
        Find the most appropriate Phase 1 finding collection.
        """

        candidate_keys = (
            "findings",
            "issues",
            "signals",
            "anomalies",
            "reasons",
        )

        for key in candidate_keys:

            value = self._get(
                integrity_result,
                key,
            )

            if value is None:
                continue

            if isinstance(
                value,
                (str, bytes),
            ):
                return [
                    value
                ]

            if isinstance(
                value,
                Mapping,
            ):
                return [
                    value
                ]

            if isinstance(
                value,
                Iterable,
            ):
                return list(
                    value
                )

        return []

    # ========================================================
    # FINDING -> SIGNAL
    # ========================================================

    def _finding_to_signal(
        self,
        finding: Any,
    ) -> RiskSignal | None:

        # ----------------------------------------------------
        # String finding
        # ----------------------------------------------------

        if isinstance(
            finding,
            str,
        ):

            text = finding.strip()

            if not text:
                return None

            severity = self._infer_severity_from_text(
                text
            )

            return RiskSignal(
                code="INTEGRITY_OBSERVATION",

                category="integrity",

                severity=severity,

                message=text,

                score=self.SEVERITY_SCORE[
                    severity
                ],

                confidence=0.70,

                evidence={
                    "raw_finding":
                        text,
                },

                source=(
                    "integrity_signal_adapter"
                ),
            )

        # ----------------------------------------------------
        # Structured finding
        # ----------------------------------------------------

        code = self._normalize_code(
            self._first(
                finding,
                (
                    "code",
                    "type",
                    "name",
                    "rule",
                    "check",
                ),
                default=(
                    "INTEGRITY_OBSERVATION"
                ),
            )
        )

        message = self._normalize_text(
            self._first(
                finding,
                (
                    "message",
                    "reason",
                    "description",
                    "detail",
                    "details",
                ),
                default=code,
            )
        )

        severity = self._normalize_severity(
            self._first(
                finding,
                (
                    "severity",
                    "level",
                    "risk_level",
                ),
            ),
            message=message,
        )

        confidence = self._to_probability(
            self._first(
                finding,
                (
                    "confidence",
                    "score_confidence",
                    "certainty",
                ),
                default=0.80,
            )
        )

        # Some Phase 1 checks may explicitly mark whether the
        # finding is suspicious / failed.
        suspicious = self._first(
            finding,
            (
                "suspicious",
                "is_suspicious",
                "failed",
                "is_failed",
            ),
            default=None,
        )

        if suspicious is False:
            severity = "info"

        score = self.SEVERITY_SCORE[
            severity
        ]

        evidence = self._mapping_copy(
            finding
        )

        return RiskSignal(
            code=code,

            category="integrity",

            severity=severity,

            message=message,

            score=score,

            confidence=confidence,

            evidence=evidence,

            source=(
                "integrity_signal_adapter"
            ),
        )

    # ========================================================
    # TOP LEVEL FALLBACKS
    # ========================================================

    def _top_level_fallback_signals(
        self,
        integrity_result: Any,
    ) -> list[RiskSignal]:
        """
        Conservative support for summary-only Phase 1 outputs.
        """

        signals: list[
            RiskSignal
        ] = []

        suspicious = self._first(
            integrity_result,
            (
                "suspicious",
                "is_suspicious",
                "integrity_failed",
                "has_integrity_issues",
            ),
            default=None,
        )

        tampered = self._first(
            integrity_result,
            (
                "tampered",
                "is_tampered",
            ),
            default=None,
        )

        extension_spoofed = self._first(
            integrity_result,
            (
                "extension_spoofed",
                "is_extension_spoofed",
                "extension_mismatch",
            ),
            default=None,
        )

        if tampered is True:

            signals.append(
                RiskSignal(
                    code="INTEGRITY_TAMPERING_FLAG",

                    category="integrity",

                    severity="critical",

                    message=(
                        "Phase 1 reported explicit "
                        "tampering evidence."
                    ),

                    score=(
                        self.SEVERITY_SCORE[
                            "critical"
                        ]
                    ),

                    confidence=0.95,

                    source=(
                        "integrity_signal_adapter"
                    ),
                )
            )

        if extension_spoofed is True:

            signals.append(
                RiskSignal(
                    code="FILE_EXTENSION_MISMATCH",

                    category="integrity",

                    severity="high",

                    message=(
                        "File extension does not match "
                        "the detected document content."
                    ),

                    score=(
                        self.SEVERITY_SCORE[
                            "high"
                        ]
                    ),

                    confidence=0.95,

                    source=(
                        "integrity_signal_adapter"
                    ),
                )
            )

        if (
            suspicious is True
            and not signals
        ):

            signals.append(
                RiskSignal(
                    code="INTEGRITY_SUSPICION_FLAG",

                    category="integrity",

                    severity="medium",

                    message=(
                        "Phase 1 reported suspicious "
                        "document integrity evidence."
                    ),

                    score=(
                        self.SEVERITY_SCORE[
                            "medium"
                        ]
                    ),

                    confidence=0.80,

                    source=(
                        "integrity_signal_adapter"
                    ),
                )
            )

        return signals

    # ========================================================
    # CONFIDENCE
    # ========================================================

    def _extract_confidence(
        self,
        integrity_result: Any,
    ) -> float:

        value = self._first(
            integrity_result,
            (
                "confidence",
                "integrity_confidence",
                "analysis_confidence",
            ),
            default=0.0,
        )

        return self._to_probability(
            value
        )

    @staticmethod
    def _average_signal_confidence(
        signals: list[RiskSignal],
    ) -> float:

        if not signals:
            return 0.0

        return sum(
            signal.confidence
            for signal
            in signals
        ) / len(
            signals
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize_severity(
        self,
        value: Any,
        message: str = "",
    ) -> str:

        if value is not None:

            severity = str(
                value
            ).strip().lower()

            aliases = {
                "informational": "info",
                "warning": "low",
                "warn": "low",
                "moderate": "medium",
                "severe": "high",
                "error": "high",
                "fatal": "critical",
            }

            severity = aliases.get(
                severity,
                severity,
            )

            if severity in self.SEVERITY_SCORE:
                return severity

        return self._infer_severity_from_text(
            message
        )

    @staticmethod
    def _infer_severity_from_text(
        text: str,
    ) -> str:
        """
        Conservative fallback only.

        We avoid converting vague observations into high-risk
        evidence without an explicit Phase 1 severity.
        """

        normalized = str(
            text
        ).lower()

        critical_terms = (
            "confirmed tamper",
            "confirmed manipulation",
            "malicious",
        )

        high_terms = (
            "extension spoof",
            "file signature mismatch",
            "content type mismatch",
        )

        medium_terms = (
            "suspicious",
            "inconsistent",
            "unexpected modification",
        )

        if any(
            term in normalized
            for term in critical_terms
        ):
            return "critical"

        if any(
            term in normalized
            for term in high_terms
        ):
            return "high"

        if any(
            term in normalized
            for term in medium_terms
        ):
            return "medium"

        return "info"

    @staticmethod
    def _normalize_code(
        value: Any,
    ) -> str:

        text = str(
            value
            if value is not None
            else "INTEGRITY_OBSERVATION"
        ).strip().upper()

        normalized = "".join(
            character
            if character.isalnum()
            else "_"
            for character
            in text
        )

        normalized = "_".join(
            part
            for part
            in normalized.split("_")
            if part
        )

        return (
            normalized
            or "INTEGRITY_OBSERVATION"
        )

    @staticmethod
    def _normalize_text(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        return str(
            value
        ).strip()

    # ========================================================
    # GENERIC ACCESS
    # ========================================================

    @staticmethod
    def _get(
        obj: Any,
        key: str,
        default: Any = None,
    ) -> Any:

        if isinstance(
            obj,
            Mapping,
        ):
            return obj.get(
                key,
                default,
            )

        return getattr(
            obj,
            key,
            default,
        )

    def _first(
        self,
        obj: Any,
        keys: tuple[str, ...],
        default: Any = None,
    ) -> Any:

        for key in keys:

            value = self._get(
                obj,
                key,
                None,
            )

            if value is not None:
                return value

        return default

    @staticmethod
    def _mapping_copy(
        value: Any,
    ) -> dict[str, Any]:

        if isinstance(
            value,
            Mapping,
        ):
            return dict(
                value
            )

        if hasattr(
            value,
            "to_dict",
        ):

            try:
                result = value.to_dict()

                if isinstance(
                    result,
                    Mapping,
                ):
                    return dict(
                        result
                    )

            except Exception:
                pass

        if hasattr(
            value,
            "__dict__",
        ):
            return dict(
                vars(
                    value
                )
            )

        return {
            "raw_finding":
                str(
                    value
                )
        }

    @staticmethod
    def _to_probability(
        value: Any,
    ) -> float:

        try:
            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        return max(
            0.0,
            min(
                number,
                1.0,
            ),
        )


integrity_signal_adapter = (
    IntegritySignalAdapter()
)