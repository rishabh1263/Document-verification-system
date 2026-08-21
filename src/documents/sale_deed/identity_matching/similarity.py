"""
similarity.py

Contains reusable similarity algorithms.
No business rules should exist here.
"""

from difflib import SequenceMatcher

from .normalizer import DataNormalizer


class SimilarityEngine:
    """
    Generic similarity engine.

    This class is document-agnostic and only compares values.
    """

    # ----------------------------------------------------
    # EXACT MATCH
    # ----------------------------------------------------

    @staticmethod
    def exact_match(a: str, b: str) -> bool:
        return a == b

    # ----------------------------------------------------
    # EXACT SIMILARITY
    # ----------------------------------------------------

    @staticmethod
    def exact_similarity(a: str, b: str) -> float:

        a = DataNormalizer.normalize_text(a)
        b = DataNormalizer.normalize_text(b)

        return 100.0 if a == b else 0.0

    # ----------------------------------------------------
    # FUZZY SIMILARITY
    # ----------------------------------------------------

    @staticmethod
    def fuzzy_similarity(a: str, b: str) -> float:

        a = DataNormalizer.normalize_text(a)
        b = DataNormalizer.normalize_text(b)

        return round(
            SequenceMatcher(
                None,
                a,
                b
            ).ratio() * 100,
            2
        )

    # ----------------------------------------------------
    # TOKEN SIMILARITY
    # ----------------------------------------------------

    @staticmethod
    def token_similarity(a: str, b: str) -> float:

        a_tokens = set(
            DataNormalizer.normalize_name(a).split()
        )

        b_tokens = set(
            DataNormalizer.normalize_name(b).split()
        )

        if not a_tokens or not b_tokens:
            return 0.0

        common = len(a_tokens.intersection(b_tokens))
        total = len(a_tokens.union(b_tokens))

        return round(
            (common / total) * 100,
            2
        )

    # ----------------------------------------------------
    # DOCUMENT NUMBER SIMILARITY
    # ----------------------------------------------------

    @staticmethod
    def numeric_similarity(a: str, b: str) -> float:
        """
        Compare Aadhaar, PAN, Passport,
        Driving Licence etc.
        """

        a = DataNormalizer.normalize_document_number(a)
        b = DataNormalizer.normalize_document_number(b)

        return 100.0 if a == b else 0.0

    # ----------------------------------------------------
    # DATE SIMILARITY
    # ----------------------------------------------------

    @staticmethod
    def date_similarity(a: str, b: str) -> float:
        """
        Compare dates after normalization.
        """

        a = DataNormalizer.normalize_date(a)
        b = DataNormalizer.normalize_date(b)

        return 100.0 if a == b else 0.0

    # ----------------------------------------------------
    # BEST NAME SIMILARITY
    # ----------------------------------------------------

    @staticmethod
    def best_name_similarity(a: str, b: str) -> float:
        """
        Returns the best score among
        Exact, Fuzzy and Token similarity.
        """

        exact = SimilarityEngine.exact_similarity(a, b)

        fuzzy = SimilarityEngine.fuzzy_similarity(a, b)

        token = SimilarityEngine.token_similarity(a, b)

        return max(
            exact,
            fuzzy,
            token
        )
