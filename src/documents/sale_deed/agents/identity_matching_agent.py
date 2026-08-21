"""
Identity Matching Agent

Responsibilities
----------------
1. Validate pipeline outputs
2. Normalize role
3. Invoke Identity Comparator
4. Return standardized response
"""

from dataclasses import asdict
from typing import Any, Dict

from src.documents.sale_deed.identity_matching.comparator import IdentityComparator


class IdentityMatchingAgent:

    def __init__(self):
        self.comparator = IdentityComparator()

    # ======================================================
    # Public API
    # ======================================================

    def compare(
        self,
        source_document: Dict[str, Any],
        target_document: Dict[str, Any],
        source_document_type: str,
        target_document_type: str,
        role: str = "buyer"
    ) -> Dict[str, Any]:

        self._validate_pipeline_output(source_document, "Source")
        self._validate_pipeline_output(target_document, "Target")

        role = self._normalize_role(role)

        report = self.comparator.compare(
            source_document=source_document,
            target_document=target_document,
            source_document_type=source_document_type,
            target_document_type=target_document_type,
            role=role
        )

        return {
            "success": True,
            "message": "Identity comparison completed successfully.",
            "data": asdict(report)
        }

    # ======================================================
    # Validation
    # ======================================================

    def _validate_pipeline_output(self, document, name):

        if not document:
            raise ValueError(f"{name} document is empty.")

        if not isinstance(document, dict):
            raise ValueError(f"{name} document must be a dictionary.")

        if "fields" not in document:
            raise ValueError(
                f"{name} document must contain extracted fields."
            )

    # ======================================================
    # Role
    # ======================================================

    def _normalize_role(self, role: str):

        if role is None:
            return "buyer"

        role = role.lower().strip()

        if role not in {"buyer", "seller"}:
            raise ValueError(
                "Role must be buyer or seller."
            )

        return role
