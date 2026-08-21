from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class MatchResult:
    field_name: str
    source_value: Any
    target_value: Any

    similarity_score: float
    matched: bool
    match_type: str

    remarks: str = ""


@dataclass
class ComparisonReport:
    overall_score: float

    decision: str

    matched_fields: int
    mismatched_fields: int

    results: List[MatchResult] = field(default_factory=list)
