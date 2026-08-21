"""
Document Detection Models
=========================

Shared models for the ITR Document Detection Framework.

Reusable for:
- Detection
- Validation
- Extraction
- Matching

Author : SBFC Document Intelligence
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ==========================================================
# ENUMS
# ==========================================================


class DocumentType(str, Enum):
    ITR = "itr"
    BANK_STATEMENT = "bank_statement"
    AADHAAR = "aadhaar"
    PAN = "pan"
    GST = "gst"
    SALARY_SLIP = "salary_slip"
    UNKNOWN = "unknown"


class DocumentMode(str, Enum):
    DIGITAL = "digital"
    SCANNED = "scanned"
    MIXED = "mixed"
    IMAGE = "image"
    UNKNOWN = "unknown"


class DetectionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"


class KeywordType(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    NEGATIVE = "negative"


# ==========================================================
# INPUT
# ==========================================================


class DetectionInput(BaseModel):
    """
    Input to Detection Engine.
    """

    file_path: str = Field(
        ...,
        description="Absolute file path"
    )


# ==========================================================
# METADATA RESULT
# ==========================================================


class MetadataResult(BaseModel):
    """
    Metadata Detector Result.
    """

    valid_file: bool = False

    is_supported: bool = False

    is_valid_pdf: bool = False

    encrypted: bool = False

    corrupted: bool = False

    extension: str = ""

    file_size: int = 0

    page_count: int = 0

    mode: DocumentMode = DocumentMode.UNKNOWN

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

    sha256: str = ""

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    processing_time_ms: float = 0.0

    reasons: List[str] = Field(
        default_factory=list
    )


# ==========================================================
# KEYWORD RESULT
# ==========================================================


class KeywordMatch(BaseModel):
    """
    Matched positive keyword.
    """

    keyword: str

    weight: int

    position: int

    type: KeywordType


class NegativeKeywordMatch(BaseModel):
    """
    Matched negative keyword.
    """

    keyword: str

    penalty: int

    position: int


class KeywordResult(BaseModel):
    """
    Keyword Detector Result.
    """

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    total_positive_score: int = 0

    total_negative_score: int = 0

    total_keywords_found: int = 0

    matched_keywords: List[KeywordMatch] = Field(
        default_factory=list
    )

    negative_keywords: List[NegativeKeywordMatch] = Field(
        default_factory=list
    )

    reasons: List[str] = Field(
        default_factory=list
    )


# ==========================================================
# LAYOUT RESULT
# ==========================================================


class LayoutResult(BaseModel):
    """
    Layout Detector Result.
    """

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    detected_sections: List[str] = Field(
        default_factory=list
    )

    reasons: List[str] = Field(
        default_factory=list
    )


# ==========================================================
# STRUCTURE RESULT
# ==========================================================


class StructureResult(BaseModel):
    """
    ITR Structure Detector Result.

    Represents structural evidence found inside the document.

    Examples of structural components:
        - assessment_year
        - pan
        - income
        - tax_computation
        - tax_paid
        - verification
        - itr_identity
        - acknowledgement

    Relationships describe how structural components
    occur in relation to one another.
    """

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    detected_components: List[str] = Field(
        default_factory=list
    )

    relationships_found: List[str] = Field(
        default_factory=list
    )

    reasons: List[str] = Field(
        default_factory=list
    )


# ==========================================================
# DETECTION EVIDENCE
# ==========================================================


class DetectionEvidence(BaseModel):
    """
    Combined detector outputs.

    Detection evidence is intentionally separated into
    independent signals:

        Metadata
        Keyword
        Layout
        Structure
    """

    metadata: MetadataResult = Field(
        default_factory=MetadataResult
    )

    keyword: KeywordResult = Field(
        default_factory=KeywordResult
    )

    layout: LayoutResult = Field(
        default_factory=LayoutResult
    )

    structure: StructureResult = Field(
        default_factory=StructureResult
    )

    raw_text_length: int = 0

    first_page_only: bool = True


# ==========================================================
# FINAL RESULT
# ==========================================================


class DetectionResult(BaseModel):
    """
    Final Detection Result.
    """

    status: DetectionStatus

    detected: bool

    filename: str = ""

    document_hash: str = ""

    version: str = "1.0.0"

    document_type: DocumentType = DocumentType.UNKNOWN

    mode: DocumentMode = DocumentMode.UNKNOWN

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    page_count: int = 0

    processing_time_ms: float = 0.0

    evidence: DetectionEvidence = Field(
        default_factory=DetectionEvidence
    )

    error: Optional[str] = None