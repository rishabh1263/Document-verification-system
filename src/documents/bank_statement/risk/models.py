"""
Bank Statement Risk Models
==========================

Shared data contracts for Phase 4:

    Risk & Anomaly Signal Engine

These models are intentionally generic and bank-independent.

Phase 4 does NOT directly decide whether a document is fraudulent.
It produces explainable risk/anomaly signals that can later be
aggregated into a final risk assessment.

Design principles
-----------------
1. Risk and uncertainty are different concepts.
2. Every risk signal should be explainable.
3. Signals should preserve supporting evidence.
4. Individual analyzers should remain independent.
5. The final risk engine should aggregate signals rather than
   duplicate analyzer logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


# ============================================================
# SEVERITY
# ============================================================


VALID_SEVERITIES = {
    "info",
    "low",
    "medium",
    "high",
    "critical",
}


# ============================================================
# RISK SIGNAL
# ============================================================


@dataclass(frozen=True)
class RiskSignal:
    """
    Generic explainable risk/anomaly signal.

    Parameters
    ----------
    code:
        Stable machine-readable signal identifier.

    category:
        High-level signal family.

        Examples:
            transaction
            balance
            statement
            extraction
            integrity

    severity:
        info / low / medium / high / critical

    message:
        Human-readable explanation.

    score:
        Risk contribution produced by this signal.

        This is NOT automatically the final statement risk score.

    confidence:
        Confidence that the signal itself is correctly identified.

    transaction_sequence:
        Optional transaction sequence associated with the signal.

    field_name:
        Optional affected field.

    expected:
        Optional expected value.

    actual:
        Optional observed value.

    evidence:
        Additional structured evidence.

    source:
        Analyzer/module that produced the signal.
    """

    code: str

    category: str

    severity: str

    message: str

    score: float = 0.0

    confidence: float = 1.0

    transaction_sequence: int | None = None

    field_name: str | None = None

    expected: Any = None

    actual: Any = None

    evidence: Mapping[str, Any] = field(
        default_factory=dict
    )

    source: str | None = None

    def __post_init__(self):

        if not self.code:
            raise ValueError(
                "RiskSignal code cannot be empty."
            )

        if not self.category:
            raise ValueError(
                "RiskSignal category cannot be empty."
            )

        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                "Invalid RiskSignal severity: "
                f"{self.severity}"
            )

        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(
                "RiskSignal confidence must be "
                "between 0 and 1."
            )

        if float(self.score) < 0:
            raise ValueError(
                "RiskSignal score cannot be negative."
            )

    def to_dict(self) -> dict[str, Any]:

        data = asdict(self)

        data["score"] = round(
            float(self.score),
            4,
        )

        data["confidence"] = round(
            float(self.confidence),
            4,
        )

        return data


# ============================================================
# TRANSACTION ANOMALY
# ============================================================


@dataclass(frozen=True)
class TransactionAnomaly:
    """
    Transaction-level anomaly produced by Phase 4 analyzers.
    """

    sequence: int | None

    anomaly_type: str

    severity: str

    message: str

    score: float

    confidence: float

    amount: float | None = None

    transaction_date: str | None = None

    description: str | None = None

    evidence: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self):

        if not self.anomaly_type:
            raise ValueError(
                "Transaction anomaly type cannot be empty."
            )

        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                "Invalid transaction anomaly severity: "
                f"{self.severity}"
            )

        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(
                "Transaction anomaly confidence must be "
                "between 0 and 1."
            )

        if float(self.score) < 0:
            raise ValueError(
                "Transaction anomaly score cannot be negative."
            )

    def to_dict(self) -> dict[str, Any]:

        data = asdict(self)

        data["score"] = round(
            float(self.score),
            4,
        )

        data["confidence"] = round(
            float(self.confidence),
            4,
        )

        return data


# ============================================================
# TRANSACTION ANOMALY SUMMARY
# ============================================================


@dataclass(frozen=True)
class TransactionAnomalySummary:
    """
    Output contract for TransactionAnomalyAnalyzer.
    """

    transaction_count: int

    analyzed_count: int

    anomaly_count: int

    unusual_amount_count: int

    duplicate_like_count: int

    burst_count: int

    anomalies: tuple[
        TransactionAnomaly,
        ...,
    ]

    signals: tuple[
        RiskSignal,
        ...,
    ]

    confidence: float

    def to_dict(self) -> dict[str, Any]:

        return {
            "transaction_count":
                self.transaction_count,

            "analyzed_count":
                self.analyzed_count,

            "anomaly_count":
                self.anomaly_count,

            "unusual_amount_count":
                self.unusual_amount_count,

            "duplicate_like_count":
                self.duplicate_like_count,

            "burst_count":
                self.burst_count,

            "confidence":
                round(
                    float(self.confidence),
                    4,
                ),

            "anomalies": [
                anomaly.to_dict()
                for anomaly
                in self.anomalies
            ],

            "signals": [
                signal.to_dict()
                for signal
                in self.signals
            ],
        }


# ============================================================
# FINAL PHASE 4 ASSESSMENT
# ============================================================


@dataclass(frozen=True)
class RiskAssessment:
    """
    Final standardized Phase 4 risk result.

    This model will be populated later by risk_aggregator.py.
    """

    risk_score: float

    risk_level: str

    assessment_confidence: float

    signal_count: int

    high_risk_signal_count: int

    critical_signal_count: int

    signals: tuple[
        RiskSignal,
        ...,
    ]

    reasons: tuple[
        str,
        ...,
    ]

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self):

        if not 0.0 <= float(self.risk_score) <= 100.0:
            raise ValueError(
                "risk_score must be between 0 and 100."
            )

        if not 0.0 <= float(
            self.assessment_confidence
        ) <= 1.0:
            raise ValueError(
                "assessment_confidence must be "
                "between 0 and 1."
            )

    def to_dict(self) -> dict[str, Any]:

        return {
            "risk_score":
                round(
                    float(self.risk_score),
                    4,
                ),

            "risk_level":
                self.risk_level,

            "assessment_confidence":
                round(
                    float(
                        self.assessment_confidence
                    ),
                    4,
                ),

            "signal_count":
                self.signal_count,

            "high_risk_signal_count":
                self.high_risk_signal_count,

            "critical_signal_count":
                self.critical_signal_count,

            "signals": [
                signal.to_dict()
                for signal
                in self.signals
            ],

            "reasons":
                list(
                    self.reasons
                ),

            "metadata":
                dict(
                    self.metadata
                ),
        }