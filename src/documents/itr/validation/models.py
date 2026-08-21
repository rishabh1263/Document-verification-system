"""
==============================================================
ITR Validation Models
==============================================================

Shared models for the ITR Validation Framework.

Validation is performed after the Detection Engine has
identified the document as an ITR.

Validation covers:

- Document integrity
- Required ITR content
- Internal consistency
- Validation evidence
- Final validation decision

This module does NOT perform:

- Document detection
- OCR
- Field extraction
- External PAN verification
- Customer matching

Author : SBFC Document Intelligence
==============================================================
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ==========================================================
# VALIDATION STATUS
# ==========================================================


class ValidationStatus(str, Enum):
    """
    Overall validation execution status.
    """

    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"


# ==========================================================
# VALIDATION DECISION
# ==========================================================


class ValidationDecision(str, Enum):
    """
    Final validation decision.
    """

    VALID = "valid"
    INVALID = "invalid"
    REVIEW = "review"
    UNKNOWN = "unknown"


# ==========================================================
# VALIDATION INPUT
# ==========================================================


class ValidationInput(BaseModel):
    """
    Input to the ITR Validation Engine.
    """

    file_path: str = Field(
        ...,
        description="Absolute path of the ITR document",
    )

    detection_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence returned by Detection Engine",
    )


# ==========================================================
# INTEGRITY RESULT
# ==========================================================


class IntegrityResult(BaseModel):
    """
    Document integrity validation result.

    This represents technical document health only.

    It does NOT determine whether the document is genuine
    or whether the document is an ITR.
    """

    valid_file: bool = False

    valid_pdf: bool = False

    readable: bool = False

    encrypted: bool = False

    corrupted: bool = False

    page_count: int = 0

    file_size: int = 0

    sha256: str = ""

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    processing_time_ms: float = 0.0

    reasons: List[str] = Field(
        default_factory=list,
    )


# ==========================================================
# CONTENT RESULT
# ==========================================================


class ContentResult(BaseModel):
    """
    Basic ITR content validation result.
    """

    required_content_present: bool = False

    assessment_year_present: bool = False

    pan_present: bool = False

    taxpayer_information_present: bool = False

    income_information_present: bool = False

    tax_computation_present: bool = False

    verification_present: bool = False

    acknowledgement_present: bool = False

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    missing_items: List[str] = Field(
        default_factory=list,
    )

    reasons: List[str] = Field(
        default_factory=list,
    )

    processing_time_ms: float = 0.0


# ==========================================================
# CONSISTENCY RESULT
# ==========================================================


class ConsistencyResult(BaseModel):
    """
    Internal consistency validation result.
    """

    consistent: bool = False

    assessment_year_consistent: bool = False

    pan_consistent: bool = False

    income_consistent: bool = False

    tax_consistent: bool = False

    acknowledgement_consistent: bool = False

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    inconsistencies: List[str] = Field(
        default_factory=list,
    )

    reasons: List[str] = Field(
        default_factory=list,
    )

    processing_time_ms: float = 0.0


# ==========================================================
# VALIDATION EVIDENCE
# ==========================================================


class ValidationEvidence(BaseModel):
    """
    Combined validation evidence.
    """

    integrity: IntegrityResult = Field(
        default_factory=IntegrityResult,
    )

    content: ContentResult = Field(
        default_factory=ContentResult,
    )

    consistency: ConsistencyResult = Field(
        default_factory=ConsistencyResult,
    )


# ==========================================================
# FINAL VALIDATION RESULT
# ==========================================================


class ValidationResult(BaseModel):
    """
    Final ITR Validation Result.
    """

    status: ValidationStatus

    decision: ValidationDecision

    valid: bool = False

    filename: str = ""

    document_hash: str = ""

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    processing_time_ms: float = 0.0

    evidence: ValidationEvidence = Field(
        default_factory=ValidationEvidence,
    )

    reasons: List[str] = Field(
        default_factory=list,
    )

    error: Optional[str] = None