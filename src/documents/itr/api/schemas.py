"""
ITR API Schemas.

Production response models for:

    Detection
        +
    Extraction
        +
    Validation
        +
    Authenticity Assessment

Important:

    validation.valid == True
        does NOT mean
    authenticity.verified == True

A document can be structurally valid and internally
consistent while still being unverified.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# VALIDATION RESPONSE
# ============================================================


class ValidationResponse(BaseModel):
    """
    Minimal validation result.
    """

    valid: bool

    decision: str

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


# ============================================================
# AUTHENTICITY SIGNAL
# ============================================================


class AuthenticitySignalResponse(BaseModel):
    """
    One explainable authenticity signal.
    """

    rule_id: str

    category: str

    severity: str

    message: str

    reason: str

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    evidence: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================
# AUTHENTICITY RESPONSE
# ============================================================


class AuthenticityResponse(BaseModel):
    """
    Final authenticity assessment.

    IMPORTANT:

        verified=True

    is only returned when an authoritative verification
    source has actually confirmed the document.

    Heuristic checks alone cannot prove genuineness.
    """

    status: str

    decision: str

    risk_level: str

    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    verified: bool

    evidence_count: int = Field(
        default=0,
        ge=0,
    )

    critical_count: int = Field(
        default=0,
        ge=0,
    )

    high_count: int = Field(
        default=0,
        ge=0,
    )

    medium_count: int = Field(
        default=0,
        ge=0,
    )

    low_count: int = Field(
        default=0,
        ge=0,
    )

    signals: list[
        AuthenticitySignalResponse
    ] = Field(
        default_factory=list,
    )

    reason: str

    summary: str


# ============================================================
# FINAL ITR RESPONSE
# ============================================================


class ITRVerifyResponse(BaseModel):
    """
    Final ITR verification API response.

    Pipeline:

        Upload
          ↓
        Detection
          ↓
        Extraction
          ↓
        Validation
          ↓
        Authenticity Scoring
          ↓
        JSON Response
    """

    success: bool

    document_type: str

    detected: bool

    detection_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    itr_form: Optional[str] = None

    name: Optional[str] = None

    pan: Optional[str] = None

    assessment_year: Optional[str] = None

    dob: Optional[str] = None

    total_income: Optional[int] = None

    business_income: Optional[int] = None

    validation: Optional[
        ValidationResponse
    ] = None

    authenticity: Optional[
        AuthenticityResponse
    ] = None

    reason: str