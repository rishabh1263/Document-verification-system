"""
matcher.py

Field Matcher

Combines normalization and similarity algorithms.
Returns standardized MatchResult objects.
"""

from .models import MatchResult
from .normalizer import DataNormalizer
from .similarity import SimilarityEngine


class FieldMatcher:
    """
    Generic field matcher.

    This class DOES NOT know anything about Sale Deed,
    Aadhaar, PAN, Passport etc.

    It only compares two values.
    """

    # Thresholds
    NAME_MATCH_THRESHOLD = 90.0
    PARTIAL_MATCH_THRESHOLD = 75.0

    ADDRESS_MATCH_THRESHOLD = 85.0

    DOCUMENT_MATCH_THRESHOLD = 100.0

    DATE_MATCH_THRESHOLD = 100.0

    # ---------------------------------------------------
    # NAME MATCHING
    # ---------------------------------------------------

    @classmethod
    def match_name(
        cls,
        field_name: str,
        source_value: str,
        target_value: str
    ) -> MatchResult:

        source = DataNormalizer.normalize_name(source_value)
        target = DataNormalizer.normalize_name(target_value)

        exact = SimilarityEngine.exact_similarity(
            source,
            target
        )

        fuzzy = SimilarityEngine.fuzzy_similarity(
            source,
            target
        )

        token = SimilarityEngine.token_similarity(
            source,
            target
        )

        score = max(
            exact,
            fuzzy,
            token
        )

        if score >= cls.NAME_MATCH_THRESHOLD:

            matched = True
            match_type = "Exact/Strong"

        elif score >= cls.PARTIAL_MATCH_THRESHOLD:

            matched = False
            match_type = "Partial"

        else:

            matched = False
            match_type = "Mismatch"

        return MatchResult(

            field_name=field_name,

            source_value=source_value,

            target_value=target_value,

            similarity_score=round(score, 2),

            matched=matched,

            match_type=match_type,

            remarks=f"Exact={exact}, Fuzzy={round(fuzzy,2)}, Token={round(token,2)}"

        )

    # ---------------------------------------------------
    # ADDRESS MATCHING
    # ---------------------------------------------------

    @classmethod
    def match_address(
        cls,
        field_name: str,
        source_value: str,
        target_value: str
    ) -> MatchResult:

        source = DataNormalizer.normalize_address(source_value)
        target = DataNormalizer.normalize_address(target_value)

        score = SimilarityEngine.fuzzy_similarity(
            source,
            target
        )

        if score >= cls.ADDRESS_MATCH_THRESHOLD:

            matched = True
            match_type = "Matched"

        elif score >= cls.PARTIAL_MATCH_THRESHOLD:

            matched = False
            match_type = "Partial"

        else:

            matched = False
            match_type = "Mismatch"

        return MatchResult(

            field_name=field_name,

            source_value=source_value,

            target_value=target_value,

            similarity_score=round(score, 2),

            matched=matched,

            match_type=match_type,

            remarks="Address comparison"

        )

    # ---------------------------------------------------
    # DOCUMENT NUMBER MATCHING
    # ---------------------------------------------------

    @classmethod
    def match_document_number(
        cls,
        field_name: str,
        source_value: str,
        target_value: str
    ) -> MatchResult:

        source = DataNormalizer.normalize_document_number(
            source_value
        )

        target = DataNormalizer.normalize_document_number(
            target_value
        )

        score = SimilarityEngine.numeric_similarity(
            source,
            target
        )

        matched = score >= cls.DOCUMENT_MATCH_THRESHOLD

        return MatchResult(

            field_name=field_name,

            source_value=source_value,

            target_value=target_value,

            similarity_score=score,

            matched=matched,

            match_type="Exact" if matched else "Mismatch",

            remarks="Document number comparison"

        )

    # ---------------------------------------------------
    # DATE MATCHING
    # ---------------------------------------------------

    @classmethod
    def match_date(
        cls,
        field_name: str,
        source_value: str,
        target_value: str
    ) -> MatchResult:

        source = DataNormalizer.normalize_date(
            source_value
        )

        target = DataNormalizer.normalize_date(
            target_value
        )

        score = SimilarityEngine.date_similarity(
            source,
            target
        )

        matched = score >= cls.DATE_MATCH_THRESHOLD

        return MatchResult(

            field_name=field_name,

            source_value=source_value,

            target_value=target_value,

            similarity_score=score,

            matched=matched,

            match_type="Exact" if matched else "Mismatch",

            remarks="Date comparison"

        )

    # ---------------------------------------------------
    # GENERIC TEXT MATCH
    # ---------------------------------------------------

    @classmethod
    def match_text(
        cls,
        field_name: str,
        source_value: str,
        target_value: str,
        threshold: float = 85.0
    ) -> MatchResult:

        source = DataNormalizer.normalize_text(source_value)
        target = DataNormalizer.normalize_text(target_value)

        score = SimilarityEngine.fuzzy_similarity(
            source,
            target
        )

        matched = score >= threshold

        if matched:

            match_type = "Matched"

        elif score >= cls.PARTIAL_MATCH_THRESHOLD:

            match_type = "Partial"

        else:

            match_type = "Mismatch"

        return MatchResult(

            field_name=field_name,

            source_value=source_value,

            target_value=target_value,

            similarity_score=round(score, 2),

            matched=matched,

            match_type=match_type,

            remarks="Generic text comparison"

        )
