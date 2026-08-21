"""
comparator.py

Main Identity Comparison Engine

Loads comparison rules dynamically and generates
a weighted comparison report.
"""

from typing import List

from .matcher import FieldMatcher
from .models import MatchResult, ComparisonReport
from .rules import (
    get_rules,
    MATCHER_NAME,
    MATCHER_ADDRESS,
    MATCHER_DOCUMENT,
    MATCHER_DATE,
    MATCHER_TEXT
)


class IdentityComparator:

    def __init__(self):
        self.results: List[MatchResult] = []

    # =====================================================
    # PUBLIC METHOD
    # =====================================================

    def compare(
        self,
        source_document: dict,
        target_document: dict,
        source_document_type: str,
        target_document_type: str,
        role: str = "buyer"
    ) -> ComparisonReport:

        self.results = []

        rules = get_rules(
            source_document_type,
            target_document_type,
            role
        )

        if not rules:

            return ComparisonReport(
                overall_score=0,
                decision="NO RULES FOUND",
                matched_fields=0,
                mismatched_fields=0,
                results=[]
            )

        weighted_score = 0
        total_weight = 0

        for rule in rules:

            source_value = source_document.get(
                rule["source_field"],
                ""
            )

            target_value = target_document.get(
                rule["target_field"],
                ""
            )

            # Handle list values (Buyer/Seller)
            if isinstance(source_value, list):

                source_value = (
                    source_value[0]
                    if source_value
                    else ""
                )

            matcher = rule["matcher"]

            # -------------------------------
            # Select matcher dynamically
            # -------------------------------

            if matcher == MATCHER_NAME:

                result = FieldMatcher.match_name(
                    rule["field_name"],
                    source_value,
                    target_value
                )

            elif matcher == MATCHER_ADDRESS:

                result = FieldMatcher.match_address(
                    rule["field_name"],
                    source_value,
                    target_value
                )

            elif matcher == MATCHER_DOCUMENT:

                result = FieldMatcher.match_document_number(
                    rule["field_name"],
                    source_value,
                    target_value
                )

            elif matcher == MATCHER_DATE:

                result = FieldMatcher.match_date(
                    rule["field_name"],
                    source_value,
                    target_value
                )

            else:

                result = FieldMatcher.match_text(
                    rule["field_name"],
                    source_value,
                    target_value
                )

            self.results.append(result)

            weighted_score += (
                result.similarity_score *
                rule["weight"]
            )

            total_weight += rule["weight"]

        overall_score = round(
            weighted_score / total_weight,
            2
        )

        matched_fields = sum(
            1
            for result in self.results
            if result.matched
        )

        mismatched_fields = (
            len(self.results) -
            matched_fields
        )

        # ---------------------------------
        # Decision
        # ---------------------------------

        if overall_score >= 95:

            decision = "VERIFIED"

        elif overall_score >= 80:

            decision = "MANUAL REVIEW"

        else:

            decision = "NOT VERIFIED"

        return ComparisonReport(

            overall_score=overall_score,

            decision=decision,

            matched_fields=matched_fields,

            mismatched_fields=mismatched_fields,

            results=self.results

        )
