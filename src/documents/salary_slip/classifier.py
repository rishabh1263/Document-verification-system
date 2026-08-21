"""
Lightweight, fast document type classifier.

Deliberately NOT a trained ML model: for a first version this keyword-based
approach is instant (no inference cost), fully transparent/debuggable, and
easy to extend (add a doc type = add a keyword list in config.py). If
volume/variety grows later, this function is the single place to swap in
a trained text classifier without touching the rest of the pipeline.
"""

from typing import Tuple

from src.documents.salary_slip import config


def classify(text: str) -> Tuple[str, float]:
    """
    Returns (doc_type, confidence) where confidence is the fraction of
    matched keywords for the winning type relative to its keyword list.
    """
    text_lower = text.lower()
    scores = {}

    for doc_type, keywords in config.DOC_TYPE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        scores[doc_type] = hits / len(keywords) if keywords else 0.0

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score < config.DOC_TYPE_MIN_CONFIDENCE:
        return "generic", best_score

    return best_type, best_score

