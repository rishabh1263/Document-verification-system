"""
Public Bank Statement Verification API Schemas.

Keeps internal forensic details separate from the concise
business-facing API response.
"""

from __future__ import annotations

from pydantic import BaseModel


class DocumentSummary(BaseModel):
    filename: str
    type: str
    confidence: float
    mode: str
    requires_ocr: bool


class VerificationSummary(BaseModel):
    decision: str

    risk_level: str | None
    risk_score: int | None

    tamper_suspected: bool | None
    authenticity: str | None

    manual_review_required: bool


class IntegritySummary(BaseModel):
    page_count: int
    pages_with_text: int

    page_consistency: float

    structural_outliers: int

    metadata_available: bool


class BankStatementVerificationResponse(BaseModel):
    status: str

    document: DocumentSummary

    verification: VerificationSummary

    integrity: IntegritySummary | None

    observations: list[str]

    message: str