"""
Pydantic Schemas for Identity Comparison API
"""

from typing import Dict, Any

from pydantic import BaseModel


class IdentityComparisonRequest(BaseModel):

    source_document: Dict[str, Any]

    target_document: Dict[str, Any]

    source_document_type: str

    target_document_type: str

    role: str = "buyer"
