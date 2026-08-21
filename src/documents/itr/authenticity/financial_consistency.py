"""
ITR financial consistency engine.

Purpose
-------
Validate relationships between extracted financial fields.

This layer does NOT decide whether an ITR is genuine by itself.

It answers a narrower question:

    "Do the financial values present in this ITR make sense
     together?"

Examples of checks:

    Business Income == Total Income

    Total Income >= Business Income
    when no deductions/other adjustments are present.

    Tax / rebate / payable relationships are checked only when
    the required fields are available.

Important
---------
Missing data is NOT treated as fraud.

A financial anomaly is evidence for the overall authenticity/risk
engine, not standalone proof that the document is fake.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


# ==========================================================
# ENUMS
# ==========================================================


class FinancialSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FinancialStatus(str, Enum):
    CLEAN = "clean"
    INCOMPLETE = "incomplete"
    SUSPICIOUS = "suspicious"


# ==========================================================
# MODELS
# ==========================================================


@dataclass(frozen=True)
class FinancialFinding:
    """
    One explainable financial consistency finding.
    """

    rule_id: str

    severity: FinancialSeverity

    message: str

    reason: str

    evidence: dict[str, Any] = field(
        default_factory=dict
    )

    score: float = 0.0


@dataclass(frozen=True)
class FinancialSnapshot:
    """
    Financial values extracted from an ITR.

    All values are optional because different ITR forms can expose
    different fields.
    """

    total_income: int | float | None = None

    business_income: int | float | None = None

    tax_on_total_income: int | float | None = None

    rebate: int | float | None = None

    net_tax_payable: int | float | None = None

    refund: int | float | None = None

    amount_payable: int | float | None = None

    deductions: int | float | None = None

    other_income: int | float | None = None


@dataclass(frozen=True)
class FinancialConsistencyResult:
    """
    Result of financial consistency analysis.
    """

    status: FinancialStatus

    risk_level: FinancialSeverity

    risk_score: float

    confidence: float

    findings: tuple[
        FinancialFinding,
        ...

    ]

    snapshot: FinancialSnapshot

    reason: str

    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,

            "risk_level": (
                self.risk_level.value
            ),

            "risk_score": self.risk_score,

            "confidence": self.confidence,

            "snapshot": asdict(
                self.snapshot
            ),

            "findings": [
                {
                    "rule_id": finding.rule_id,

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
# NUMBER NORMALIZATION
# ==========================================================


def normalize_amount(
    value: Any,
) -> float | None:
    """
    Normalize an extracted Indian-format monetary value.

    Supported examples:

        498000
        "498000"
        "4,98,000"
        "₹4,98,000"
        "Rs. 4,98,000"

    Parentheses are treated as negative values:

        "(1,000)" -> -1000
    """

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return None

    if isinstance(
        value,
        (int, float),
    ):

        return float(
            value
        )

    text = str(
        value
    ).strip()

    if not text:
        return None

    negative = (
        text.startswith("(")
        and
        text.endswith(")")
    )

    text = text.replace(
        ",",
        "",
    )

    text = text.replace(
        "₹",
        "",
    )

    text = re.sub(
        r"\bRs\.?\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.strip(
        " ()"
    )

    # Remove accidental OCR spaces.
    text = re.sub(
        r"\s+",
        "",
        text,
    )

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if not match:
        return None

    try:
        number = float(
            match.group(0)
        )
    except ValueError:
        return None

    if negative:
        number = -abs(
            number
        )

    return number


# ==========================================================
# ENGINE
# ==========================================================


class FinancialConsistencyEngine:
    """
    Analyze financial relationships in an ITR.
    """

    # ======================================================
    # PUBLIC API
    # ======================================================

    def analyze(
        self,
        snapshot: FinancialSnapshot,
    ) -> FinancialConsistencyResult:
        """
        Run all available financial consistency rules.
        """

        findings: list[
            FinancialFinding
        ] = []

        findings.extend(
            self._check_business_income_vs_total_income(
                snapshot
            )
        )

        findings.extend(
            self._check_total_income_components(
                snapshot
            )
        )

        findings.extend(
            self._check_tax_and_rebate(
                snapshot
            )
        )

        findings.extend(
            self._check_payable_relationship(
                snapshot
            )
        )

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
                snapshot,
                findings,
            )
        )

        reason = (
            self._reason(
                snapshot,
                findings,
                status,
            )
        )

        summary = (
            self._summary(
                status
            )
        )

        confidence = (
            self._confidence(
                findings,
                risk_level,
            )
        )

        return FinancialConsistencyResult(
            status=status,

            risk_level=risk_level,

            risk_score=risk_score,

            confidence=confidence,

            findings=tuple(
                findings
            ),

            snapshot=snapshot,

            reason=reason,

            summary=summary,
        )

    # ======================================================
    # BUSINESS INCOME
    # ======================================================

    @staticmethod
    def _check_business_income_vs_total_income(
        snapshot: FinancialSnapshot,
    ) -> list[
        FinancialFinding
    ]:
        """
        Check the common ITR-4 presumptive-business scenario.

        If business income is greater than total income and there
        are no known deductions/adjustments, flag it.

        If business income and total income are equal, this is
        consistent and no finding is created.
        """

        business = normalize_amount(
            snapshot.business_income
        )

        total = normalize_amount(
            snapshot.total_income
        )

        if (
            business is None
            or
            total is None
        ):
            return []

        if (
            business < 0
            or
            total < 0
        ):
            return []

        if (
            business
            ==
            total
        ):
            return []

        deductions = normalize_amount(
            snapshot.deductions
        )

        other_income = normalize_amount(
            snapshot.other_income
        )

        # --------------------------------------------------
        # Business income greater than total income can be
        # legitimate when deductions/adjustments exist.
        # --------------------------------------------------

        if (
            business
            >
            total
            and
            deductions
            in {
                None,
                0.0,
            }
        ):

            return [
                FinancialFinding(
                    rule_id=(
                        "BUSINESS_INCOME_EXCEEDS_TOTAL_INCOME"
                    ),

                    severity=(
                        FinancialSeverity.HIGH
                    ),

                    message=(
                        "Business income exceeds total income "
                        "without an identified deduction."
                    ),

                    reason=(
                        "The submitted ITR reports business income "
                        "higher than total income, while no deduction "
                        "or adjustment was available to explain the "
                        "difference. This creates a financial "
                        "consistency anomaly."
                    ),

                    evidence={
                        "business_income": business,

                        "total_income": total,

                        "deductions": deductions,

                        "other_income": other_income,
                    },

                    score=65.0,
                )
            ]

        return []

    # ======================================================
    # TOTAL INCOME COMPONENTS
    # ======================================================

    @staticmethod
    def _check_total_income_components(
        snapshot: FinancialSnapshot,
    ) -> list[
        FinancialFinding
    ]:
        """
        Check the relationship:

            business income
            + other income
            - deductions
            ≈ total income

        This rule only activates when all required values exist.
        """

        business = normalize_amount(
            snapshot.business_income
        )

        other = normalize_amount(
            snapshot.other_income
        )

        deductions = normalize_amount(
            snapshot.deductions
        )

        total = normalize_amount(
            snapshot.total_income
        )

        if (
            business is None
            or
            other is None
            or
            deductions is None
            or
            total is None
        ):
            return []

        expected = (
            business
            +
            other
            -
            deductions
        )

        difference = abs(
            expected
            -
            total
        )

        # Allow a small rounding tolerance.
        tolerance = max(
            1.0,
            abs(total) * 0.005,
        )

        if difference > tolerance:

            return [
                FinancialFinding(
                    rule_id=(
                        "TOTAL_INCOME_COMPONENT_MISMATCH"
                    ),

                    severity=(
                        FinancialSeverity.HIGH
                    ),

                    message=(
                        "Total income does not reconcile "
                        "with the available income components."
                    ),

                    reason=(
                        "The reported total income differs materially "
                        "from the calculated total using the available "
                        "business income, other income, and deductions."
                    ),

                    evidence={
                        "business_income": business,

                        "other_income": other,

                        "deductions": deductions,

                        "reported_total_income": total,

                        "calculated_total_income": (
                            expected
                        ),

                        "difference": difference,

                        "tolerance": tolerance,
                    },

                    score=60.0,
                )
            ]

        return []

    # ======================================================
    # TAX / REBATE
    # ======================================================

    @staticmethod
    def _check_tax_and_rebate(
        snapshot: FinancialSnapshot,
    ) -> list[
        FinancialFinding
    ]:
        """
        Validate basic tax/rebate relationship.

        Rebate should not exceed tax on total income.

        This is a structural consistency check only.

        Actual tax liability depends on assessment year,
        applicable regime, taxpayer status, age, and other rules.
        """

        tax = normalize_amount(
            snapshot.tax_on_total_income
        )

        rebate = normalize_amount(
            snapshot.rebate
        )

        if (
            tax is None
            or
            rebate is None
        ):
            return []

        if (
            tax < 0
            or
            rebate < 0
        ):
            return []

        if rebate > tax:

            return [
                FinancialFinding(
                    rule_id=(
                        "REBATE_EXCEEDS_TAX"
                    ),

                    severity=(
                        FinancialSeverity.HIGH
                    ),

                    message=(
                        "Reported rebate exceeds "
                        "reported tax on total income."
                    ),

                    reason=(
                        "The rebate amount is greater than the "
                        "reported tax before rebate. This violates "
                        "a basic financial relationship and should "
                        "be investigated."
                    ),

                    evidence={
                        "tax_on_total_income": tax,

                        "rebate": rebate,
                    },

                    score=65.0,
                )
            ]

        return []

    # ======================================================
    # PAYABLE RELATIONSHIP
    # ======================================================

    @staticmethod
    def _check_payable_relationship(
        snapshot: FinancialSnapshot,
    ) -> list[
        FinancialFinding
    ]:
        """
        Perform a conservative payable/refund sanity check.

        This intentionally does NOT calculate Indian tax liability
        because tax calculation requires assessment-year-specific
        rules.

        It only checks impossible sign relationships.
        """

        payable = normalize_amount(
            snapshot.amount_payable
        )

        net_payable = normalize_amount(
            snapshot.net_tax_payable
        )

        refund = normalize_amount(
            snapshot.refund
        )

        findings: list[
            FinancialFinding
        ] = []

        if (
            payable is not None
            and
            payable < 0
        ):

            findings.append(
                FinancialFinding(
                    rule_id=(
                        "NEGATIVE_AMOUNT_PAYABLE"
                    ),

                    severity=(
                        FinancialSeverity.MEDIUM
                    ),

                    message=(
                        "Amount payable contains "
                        "an unexpected negative value."
                    ),

                    reason=(
                        "The amount payable field contains a negative "
                        "value. A refund should normally be represented "
                        "separately rather than as a negative payable."
                    ),

                    evidence={
                        "amount_payable": payable,
                    },

                    score=35.0,
                )
            )

        if (
            net_payable is not None
            and
            net_payable < 0
        ):

            findings.append(
                FinancialFinding(
                    rule_id=(
                        "NEGATIVE_NET_TAX_PAYABLE"
                    ),

                    severity=(
                        FinancialSeverity.MEDIUM
                    ),

                    message=(
                        "Net tax payable contains "
                        "an unexpected negative value."
                    ),

                    reason=(
                        "A negative net payable should normally be "
                        "represented as a refund rather than as a "
                        "negative payable amount."
                    ),

                    evidence={
                        "net_tax_payable": (
                            net_payable
                        ),
                    },

                    score=35.0,
                )
            )

        if (
            refund is not None
            and
            refund < 0
        ):

            findings.append(
                FinancialFinding(
                    rule_id=(
                        "NEGATIVE_REFUND"
                    ),

                    severity=(
                        FinancialSeverity.MEDIUM
                    ),

                    message=(
                        "Refund contains an unexpected "
                        "negative value."
                    ),

                    reason=(
                        "The refund field contains a negative amount, "
                        "which creates a basic financial consistency "
                        "anomaly."
                    ),

                    evidence={
                        "refund": refund,
                    },

                    score=35.0,
                )
            )

        return findings

    # ======================================================
    # RISK
    # ======================================================

    @staticmethod
    def _calculate_risk(
        findings: list[
            FinancialFinding
        ],
    ) -> float:

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
                for value in scores[1:]
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
            FinancialFinding
        ],

        risk_score: float,
    ) -> FinancialSeverity:

        severities = [
            finding.severity
            for finding in findings
        ]

        if (
            FinancialSeverity.CRITICAL
            in severities
        ):
            return FinancialSeverity.CRITICAL

        if (
            FinancialSeverity.HIGH
            in severities
        ):
            return FinancialSeverity.HIGH

        if (
            FinancialSeverity.MEDIUM
            in severities
        ):
            return FinancialSeverity.MEDIUM

        if (
            FinancialSeverity.LOW
            in severities
        ):
            return FinancialSeverity.LOW

        if risk_score >= 0.75:
            return FinancialSeverity.CRITICAL

        if risk_score >= 0.50:
            return FinancialSeverity.HIGH

        if risk_score >= 0.25:
            return FinancialSeverity.MEDIUM

        if risk_score > 0:
            return FinancialSeverity.LOW

        return FinancialSeverity.INFO

    # ======================================================
    # STATUS
    # ======================================================

    @staticmethod
    def _status(
        snapshot: FinancialSnapshot,

        findings: list[
            FinancialFinding
        ],
    ) -> FinancialStatus:

        values = (
            snapshot.total_income,
            snapshot.business_income,
            snapshot.tax_on_total_income,
        )

        available = sum(
            value is not None
            for value in values
        )

        if not findings:

            if available == 0:
                return FinancialStatus.INCOMPLETE

            return FinancialStatus.CLEAN

        return FinancialStatus.SUSPICIOUS

    # ======================================================
    # REASON
    # ======================================================

    @staticmethod
    def _reason(
        snapshot: FinancialSnapshot,

        findings: list[
            FinancialFinding
        ],

        status: FinancialStatus,
    ) -> str:

        if not findings:

            if (
                status
                == FinancialStatus.INCOMPLETE
            ):

                return (
                    "Insufficient financial fields were available "
                    "for a complete consistency analysis."
                )

            return (
                "The available financial values passed the "
                "implemented consistency checks."
            )

        primary = max(
            findings,
            key=lambda finding: (
                finding.severity.value,
                finding.score,
            ),
        )

        return (
            primary.reason
            or
            primary.message
        )

    # ======================================================
    # SUMMARY
    # ======================================================

    @staticmethod
    def _summary(
        status: FinancialStatus,
    ) -> str:

        if (
            status
            == FinancialStatus.CLEAN
        ):
            return (
                "Financial consistency checks passed."
            )

        if (
            status
            == FinancialStatus.INCOMPLETE
        ):
            return (
                "Financial data was insufficient "
                "for complete consistency analysis."
            )

        return (
            "Financial consistency anomalies were detected."
        )

    # ======================================================
    # CONFIDENCE
    # ======================================================

    @staticmethod
    def _confidence(
        findings: list[
            FinancialFinding
        ],

        risk_level: FinancialSeverity,
    ) -> float:

        if (
            risk_level
            == FinancialSeverity.CRITICAL
        ):
            return 0.95

        if (
            risk_level
            == FinancialSeverity.HIGH
        ):
            return 0.90

        if (
            risk_level
            == FinancialSeverity.MEDIUM
        ):
            return 0.75

        if (
            risk_level
            == FinancialSeverity.LOW
        ):
            return 0.60

        return 0.55


# ==========================================================
# CONVENIENCE FUNCTION
# ==========================================================


def analyze_financial_consistency(
    snapshot: FinancialSnapshot,
) -> FinancialConsistencyResult:
    """
    Convenience wrapper.
    """

    return (
        FinancialConsistencyEngine().analyze(
            snapshot
        )
    )