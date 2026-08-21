"""
Common authoritative-verification interface.

Purpose
-------
Keep "local document authenticity" separate from "authoritative verification".

Local checks can establish that an uploaded document:
    - looks like the expected document,
    - contains coherent fields,
    - passes OCR/extraction checks,
    - has no strong local tamper indicators.

They cannot, by themselves, prove that the document was actually issued by
the authority.

This module provides a common interface for PAN, Aadhaar, Voter ID,
passbooks, etc. without hardcoding a government API that the project does
not currently have.

Current state
-------------
No external authority/database is called by this module.

Therefore the default result is:

    status = NOT_PERFORMED

Later an approved provider/connector can implement the same interface and
return MATCH / MISMATCH / UNAVAILABLE / ERROR.

Important
---------
Do not treat NOT_PERFORMED as MATCH.
Do not treat local DOCUMENT_PASS as proof of government-record authenticity.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Mapping


# ============================================================================
# ENUMS
# ============================================================================

class AuthorityStatus(str, Enum):
    """Status of authoritative verification."""

    NOT_PERFORMED = "NOT_PERFORMED"
    PENDING = "PENDING"
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class AuthorityRequest:
    """
    Normalized identity information sent to an authority provider.

    Only fields available from the document should be populated.
    """

    document_type: str
    document_number: str | None = None
    name: str | None = None
    father_name: str | None = None
    dob: str | None = None


@dataclass
class AuthorityResult:
    """
    Standard result returned by every authority provider.
    """

    status: str
    provider: str = "NONE"
    document_type: str | None = None
    document_number: str | None = None

    record_found: bool | None = None
    active: bool | None = None

    name_match: bool | None = None
    father_name_match: bool | None = None
    dob_match: bool | None = None

    error: str | None = None
    message: str | None = None

    # Keep raw provider output optional and controlled.
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================================
# PROVIDER INTERFACE
# ============================================================================

class AuthorityProvider:
    """
    Base interface for an authoritative verification provider.

    A real provider should subclass this class and implement verify().
    """

    provider_name = "BASE"

    def verify(
        self,
        request: AuthorityRequest,
    ) -> AuthorityResult:
        raise NotImplementedError(
            "AuthorityProvider.verify() must be implemented "
            "by a concrete provider."
        )


class NoAuthorityProvider(AuthorityProvider):
    """
    Default provider used until an approved authority/database integration
    is available.

    It deliberately performs NO external verification.
    """

    provider_name = "NONE"

    def verify(
        self,
        request: AuthorityRequest,
    ) -> AuthorityResult:

        return AuthorityResult(
            status=AuthorityStatus.NOT_PERFORMED.value,
            provider=self.provider_name,
            document_type=request.document_type,
            document_number=request.document_number,
            record_found=None,
            active=None,
            name_match=None,
            father_name_match=None,
            dob_match=None,
            message=(
                "Authoritative verification is not configured. "
                "Only local document authenticity checks were performed."
            ),
        )


# ============================================================================
# REQUEST BUILDERS
# ============================================================================

def build_authority_request(
    *,
    document_type: str,
    document_number: str | None = None,
    name: str | None = None,
    father_name: str | None = None,
    dob: str | None = None,
) -> AuthorityRequest:
    """
    Build a normalized request for any document type.
    """

    return AuthorityRequest(
        document_type=str(
            document_type
        ).strip().lower(),

        document_number=(
            str(document_number).strip()
            if document_number
            else None
        ),

        name=(
            str(name).strip()
            if name
            else None
        ),

        father_name=(
            str(father_name).strip()
            if father_name
            else None
        ),

        dob=(
            str(dob).strip()
            if dob
            else None
        ),
    )


# ============================================================================
# PROVIDER REGISTRY
# ============================================================================

_PROVIDERS: dict[str, AuthorityProvider] = {
    "NONE": NoAuthorityProvider(),
}


def register_provider(
    provider: AuthorityProvider,
) -> None:
    """
    Register an authority provider.

    Example later:

        register_provider(MyGovernmentProvider())
    """

    if not isinstance(
        provider,
        AuthorityProvider,
    ):
        raise TypeError(
            "provider must inherit from AuthorityProvider"
        )

    name = str(
        provider.provider_name
    ).strip().upper()

    if not name:
        raise ValueError(
            "Provider must have a non-empty provider_name."
        )

    _PROVIDERS[name] = provider


def get_provider(
    provider_name: str | None = None,
) -> AuthorityProvider:
    """
    Get a registered provider.

    Defaults to NONE, which performs no external verification.
    """

    name = (
        str(provider_name).strip().upper()
        if provider_name
        else "NONE"
    )

    return _PROVIDERS.get(
        name,
        _PROVIDERS["NONE"],
    )


# ============================================================================
# RESULT NORMALIZATION
# ============================================================================

def normalize_authority_result(
    result: AuthorityResult | Mapping[str, Any],
) -> AuthorityResult:
    """
    Normalize provider output into AuthorityResult.
    """

    if isinstance(
        result,
        AuthorityResult,
    ):
        return result

    if not isinstance(
        result,
        Mapping,
    ):
        raise TypeError(
            "Authority provider must return AuthorityResult "
            "or a mapping."
        )

    status = str(
        result.get(
            "status",
            AuthorityStatus.ERROR.value,
        )
    ).upper()

    allowed = {
        item.value
        for item in AuthorityStatus
    }

    if status not in allowed:
        status = AuthorityStatus.ERROR.value

    return AuthorityResult(
        status=status,
        provider=str(
            result.get(
                "provider",
                "UNKNOWN",
            )
        ),
        document_type=result.get(
            "document_type"
        ),
        document_number=result.get(
            "document_number"
        ),
        record_found=result.get(
            "record_found"
        ),
        active=result.get(
            "active"
        ),
        name_match=result.get(
            "name_match"
        ),
        father_name_match=result.get(
            "father_name_match"
        ),
        dob_match=result.get(
            "dob_match"
        ),
        error=result.get(
            "error"
        ),
        message=result.get(
            "message"
        ),
        raw=(
            dict(result.get("raw"))
            if isinstance(
                result.get("raw"),
                Mapping,
            )
            else None
        ),
    )


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_authoritatively(
    request: AuthorityRequest,
    *,
    provider_name: str = "NONE",
) -> AuthorityResult:
    """
    Run authoritative verification through the selected provider.

    With the current default provider this returns NOT_PERFORMED.

    This function deliberately does not fabricate a government result.
    """

    provider = get_provider(
        provider_name
    )

    try:
        result = provider.verify(
            request
        )

    except Exception as exc:
        return AuthorityResult(
            status=AuthorityStatus.ERROR.value,
            provider=provider.provider_name,
            document_type=request.document_type,
            document_number=request.document_number,
            error=str(exc),
            message=(
                "Authority provider failed."
            ),
        )

    normalized = normalize_authority_result(
        result
    )

    if not normalized.provider:
        normalized.provider = (
            provider.provider_name
        )

    if not normalized.document_type:
        normalized.document_type = (
            request.document_type
        )

    if not normalized.document_number:
        normalized.document_number = (
            request.document_number
        )

    return normalized


# ============================================================================
# FINAL STATUS COMBINER
# ============================================================================

def combine_local_and_authoritative(
    *,
    local_decision: str,
    local_score: float | int | None = None,
    authority_result: AuthorityResult | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Combine local authenticity and authoritative verification.

    Rules
    -----
    Local DOCUMENT_SUSPICIOUS:
        overall = DOCUMENT_SUSPICIOUS

    Authority MISMATCH:
        overall = DOCUMENT_SUSPICIOUS

    Local PASS + authority MATCH:
        overall = VERIFIED

    Local PASS + authority NOT_PERFORMED:
        overall = DOCUMENT_PASS_PENDING_AUTHORITY

    Local REVIEW:
        overall = MANUAL_REVIEW

    This prevents a local OCR/layout pass from being mislabeled as
    government-record verification.
    """

    local = str(
        local_decision or "MANUAL_REVIEW"
    ).upper().strip()

    if authority_result is None:
        authority = AuthorityResult(
            status=AuthorityStatus.NOT_PERFORMED.value,
            provider="NONE",
        )
    else:
        authority = normalize_authority_result(
            authority_result
        )

    if authority.status == AuthorityStatus.MISMATCH:
        overall = "DOCUMENT_SUSPICIOUS"
        reason = (
            "Authoritative verification did not match "
            "the extracted document identity."
        )

    elif local == "DOCUMENT_SUSPICIOUS":
        overall = "DOCUMENT_SUSPICIOUS"
        reason = (
            "Local document authenticity checks found "
            "strong risk indicators."
        )

    elif local == "MANUAL_REVIEW":
        overall = "MANUAL_REVIEW"
        reason = (
            "Local document evidence requires manual review."
        )

    elif (
        local == "DOCUMENT_PASS"
        and authority.status == AuthorityStatus.MATCH
    ):
        overall = "VERIFIED"
        reason = (
            "Local document checks passed and authoritative "
            "verification matched."
        )

    elif local == "DOCUMENT_PASS":
        overall = "DOCUMENT_PASS_PENDING_AUTHORITY"
        reason = (
            "Local document checks passed, but authoritative "
            "verification has not confirmed the record."
        )

    else:
        overall = "MANUAL_REVIEW"
        reason = (
            "The verification state could not be safely "
            "classified as verified."
        )

    return {
        "overall_status": overall,
        "local_verification": {
            "decision": local,
            "score": (
                None
                if local_score is None
                else float(local_score)
            ),
        },
        "authoritative_verification": authority.to_dict(),
        "reason": reason,
    }


# ============================================================================
# DOCUMENT-AGNOSTIC CONVENIENCE API
# ============================================================================

def verify_document_authority(
    *,
    document_type: str,
    document_number: str | None = None,
    name: str | None = None,
    father_name: str | None = None,
    dob: str | None = None,
    provider_name: str = "NONE",
) -> dict[str, Any]:
    """
    One-call API for document validators.

    Current result will be NOT_PERFORMED unless a real provider is registered.
    """

    request = build_authority_request(
        document_type=document_type,
        document_number=document_number,
        name=name,
        father_name=father_name,
        dob=dob,
    )

    result = verify_authoritatively(
        request,
        provider_name=provider_name,
    )

    return result.to_dict()


# ============================================================================
# TESTS
# ============================================================================

def module_test() -> dict[str, Any]:
    """
    Deterministic tests.

    No network/API/database call is made.
    """

    request = build_authority_request(
        document_type="PAN",
        document_number="HTDPP2441D",
        name="Amit Akhilesh Pandey",
        father_name="Akhilesh Kumar Hiralal Pandey",
        dob="07/06/2004",
    )

    not_performed = verify_authoritatively(
        request
    )

    local_pass = combine_local_and_authoritative(
        local_decision="DOCUMENT_PASS",
        local_score=100,
        authority_result=not_performed,
    )

    local_review = combine_local_and_authoritative(
        local_decision="MANUAL_REVIEW",
        local_score=65,
        authority_result=not_performed,
    )

    local_suspicious = combine_local_and_authoritative(
        local_decision="DOCUMENT_SUSPICIOUS",
        local_score=30,
        authority_result=not_performed,
    )

    authoritative_match = AuthorityResult(
        status=AuthorityStatus.MATCH.value,
        provider="TEST_PROVIDER",
        document_type="pan",
        document_number="HTDPP2441D",
        record_found=True,
        active=True,
        name_match=True,
        father_name_match=True,
        dob_match=True,
    )

    verified = combine_local_and_authoritative(
        local_decision="DOCUMENT_PASS",
        local_score=100,
        authority_result=authoritative_match,
    )

    authoritative_mismatch = AuthorityResult(
        status=AuthorityStatus.MISMATCH.value,
        provider="TEST_PROVIDER",
        document_type="pan",
        document_number="HTDPP2441D",
        record_found=True,
        active=True,
        name_match=False,
        father_name_match=False,
        dob_match=False,
    )

    mismatch = combine_local_and_authoritative(
        local_decision="DOCUMENT_PASS",
        local_score=100,
        authority_result=authoritative_mismatch,
    )

    passed = (
        not_performed.status
        == AuthorityStatus.NOT_PERFORMED.value
        and local_pass["overall_status"]
        == "DOCUMENT_PASS_PENDING_AUTHORITY"
        and local_review["overall_status"]
        == "MANUAL_REVIEW"
        and local_suspicious["overall_status"]
        == "DOCUMENT_SUSPICIOUS"
        and verified["overall_status"]
        == "VERIFIED"
        and mismatch["overall_status"]
        == "DOCUMENT_SUSPICIOUS"
    )

    return {
        "passed": passed,
        "not_performed": not_performed.to_dict(),
        "local_pass": local_pass,
        "local_review": local_review,
        "local_suspicious": local_suspicious,
        "verified": verified,
        "mismatch": mismatch,
    }


# ============================================================================
# CONTRACT
# ============================================================================

def api_contract_test() -> dict[str, Any]:
    return {
        "default_provider": "NONE",
        "default_status": AuthorityStatus.NOT_PERFORMED.value,
        "network_call_performed": False,
        "document_agnostic": True,
        "local_and_authoritative_separated": True,
        "not_performed_is_not_verified": True,
        "supports_match": True,
        "supports_mismatch": True,
        "supports_manual_review": True,
    }


__all__ = [
    "AuthorityStatus",
    "AuthorityRequest",
    "AuthorityResult",
    "AuthorityProvider",
    "NoAuthorityProvider",
    "build_authority_request",
    "register_provider",
    "get_provider",
    "normalize_authority_result",
    "verify_authoritatively",
    "combine_local_and_authoritative",
    "verify_document_authority",
    "module_test",
    "api_contract_test",
]