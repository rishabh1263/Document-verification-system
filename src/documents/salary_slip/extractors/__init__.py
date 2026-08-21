from typing import Tuple

from src.documents.salary_slip import config
from .salary_slip import SalarySlipExtractor
from .bank_statement import BankStatementExtractor
from .id_proof import IDProofExtractor
from .generic import GenericExtractor

REGISTRY = {
    "salary_slip": SalarySlipExtractor,
    "bank_statement": BankStatementExtractor,
    "id_proof": IDProofExtractor,
    "generic": GenericExtractor,
}


def get_extractor(doc_type: str):
    cls = REGISTRY.get(doc_type, GenericExtractor)
    return cls()


def classify_doc_type(text: str) -> Tuple[str, float]:
    """
    Fast keyword-overlap classifier â€” no model to load, instant, and easy
    to extend by editing config.DOC_TYPE_KEYWORDS. Returns (doc_type,
    confidence 0-1); falls back to "generic" below DOC_TYPE_MIN_CONFIDENCE.
    """
    text_lower = text.lower()
    best_type, best_score = "generic", 0.0

    for doc_type, keywords in config.DOC_TYPE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        score = hits / len(keywords) if keywords else 0.0
        if score > best_score:
            best_type, best_score = doc_type, score

    if best_score < config.DOC_TYPE_MIN_CONFIDENCE:
        return "generic", best_score
    return best_type, best_score

