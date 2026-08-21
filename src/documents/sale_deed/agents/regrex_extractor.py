"""
regex_extractor.py â€” Production Ready (single-file build)

Context-based extraction engine for property-deed OCR text, combining:
  - DocumentExtractor   (deed/registration metadata + document type)
  - FinancialExtractor  (stamp duty, fees, sale consideration, market value)
  - PropertyExtractor   (district/village/tehsil, survey/plot/khasra/khata, area)
  - PartyExtractor      (seller/buyer/witness name, relation, address)
  - ExtractionEngine     (orchestrates all four, legacy-compatible output shape)
  - ExtractionAgentAdapter (backfill-only integration point for an LLM-first agent)

Design, and why it replaces the old positional regex:
------------------------------------------------------
The legacy approach used a single regex per field that hard-coded the
label immediately followed by the value: `Deed\\s*No\\.?\\s*[:\\-]?\\s*(...)`.
That breaks constantly on real OCR text, where the label and value are
frequently split across lines (columnar scans, wrapped tables), have
stray characters between them, or have the label appear multiple times
with only one occurrence actually followed by a value.

Here, extraction is three separate steps:
  1. Anchor search: find every place a field's label (in any supported
     language) appears in the text, regardless of what follows it.
  2. Windowing: for each anchor, build a small context window -- the
     rest of that line, and (only if the anchor is genuinely line-initial
     and its line ends right there, meaning OCR likely wrapped the value
     onto the next line) a short lookahead into subsequent lines.
  3. Value parsing: a *type* parser (alnum code / date / currency /
     area / place name) looks for a value of the right shape inside
     that window, independent of the label.

This means the same value parser can serve many differently-worded
labels, multiple anchors can be tried per field, and a value can still
be found even when OCR noise sits between the label and the value --
while a label-like keyword that happens to appear mid-sentence (e.g.
"...the total consideration") can't accidentally grab an unrelated
value off the next line.

Backward compatibility:
  ExtractionEngine.extract(text) returns the same section/field shape
  as the original RegexExtractor.extract(): document_details / financial
  / property, each with a "_confidence" key -- plus a new "parties"
  section. ExtractionEngine.extract_sale_consideration_candidates(text)
  is preserved with the same signature and return shape as before.
"""

import re
from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Dict, Callable


# ============================================================================
# Value parsers -- pure value-pattern matchers, no knowledge of labels/anchors
# ============================================================================

CURRENCY_PREFIX = r"(?:Rs\.?|INR|â‚¹|à¤°à¥\.?|à¤°à¥à¤ªà¤¯à¥‡)?\s*"

_ALNUM_RE = re.compile(r"[:\-â€“â€”]?\s*([A-Za-z0-9][A-Za-z0-9\-/]{0,24})")
_DATE_RE = re.compile(r"([0-9]{1,2}[/\-.][0-9]{1,2}[/\-.][0-9]{2,4})")
_CURRENCY_RE = re.compile(CURRENCY_PREFIX + r"([0-9][0-9,]{1,15}(?:\.[0-9]+)?)")

_AREA_UNITS = (
    r"(?:sq\.?\s*ft|sq\.?\s*yd|sq\.?\s*m(?:tr)?|acre|bigha|kattha|decimal|"
    r"hectare|ha|à¤µà¤°à¥à¤—à¤«à¥à¤Ÿ|à¤µà¤°à¥à¤—à¤®à¥€à¤Ÿà¤°|à¤à¤•à¤¡à¤¼|à¤¬à¥€à¤˜à¤¾|à¤•à¤Ÿà¥à¤ à¤¾|à¤¡à¥‡à¤¸à¥€à¤®à¤²|à¤¹à¥‡à¤•à¥à¤Ÿà¥‡à¤¯à¤°)"
)
_AREA_RE = re.compile(
    r"[:\-â€“â€”]?\s*([0-9][0-9\.,]*\s*" + _AREA_UNITS + r")", re.IGNORECASE
)

_PLACE_STOP_WORDS = [
    "district", "village", "tehsil", "taluka", "sub-district", "subdistrict",
    "survey", "plot", "khasra", "khata", "registration", "book", "volume",
    "à¤œà¤¿à¤²à¤¾", "à¤—à¤¾à¤à¤µ", "à¤—à¤¾à¤‚à¤µ", "à¤¤à¤¹à¤¸à¥€à¤²", "à¤¤à¤¾à¤²à¥à¤•à¤¾", "à¤¸à¤°à¥à¤µà¥‡", "à¤ªà¥à¤²à¥‰à¤Ÿ", "à¤–à¤¸à¤°à¤¾", "à¤–à¤¾à¤¤à¤¾",
]
_PLACE_STOP_PATTERN = "|".join(_PLACE_STOP_WORDS)
_PLACE_RE = re.compile(
    r"[:\-â€“â€”]?\s*([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s]{1,40}?)"
    r"(?:\b(?:" + _PLACE_STOP_PATTERN + r")\b|[,\n]|$)",
    re.IGNORECASE,
)


def parse_alnum_code(window: str) -> list:
    """Deed/token/registration/book/volume/page/survey/plot/khasra/khata
    style codes: alphanumeric, optionally with hyphens or slashes."""
    m = _ALNUM_RE.search(window)
    return [m.group(1)] if m else []


def parse_date(window: str) -> list:
    """dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy in 2 or 4 digit years."""
    m = _DATE_RE.search(window)
    return [m.group(1)] if m else []


def parse_currency(window: str) -> list:
    """Amounts, with or without a currency symbol/prefix."""
    m = _CURRENCY_RE.search(window)
    return [m.group(1)] if m else []


def parse_area(window: str) -> list:
    """Numeric area figure with a recognized unit attached."""
    m = _AREA_RE.search(window)
    return [m.group(1).strip()] if m else []


def parse_place_name(window: str) -> list:
    """District/village/tehsil style free-text place names, cut off at
    the next field's label or a line/comma boundary."""
    m = _PLACE_RE.search(window)
    return [m.group(1).strip()] if m else []


# ============================================================================
# Base extractor -- anchor search, context windowing, scoring, dedup
# ============================================================================

@dataclass
class FieldCandidate:
    field_name: str
    value: str
    anchor_keyword: str
    distance: int          # chars between anchor end and matched value
    line_offset: int       # 0 = same line as anchor, 1 = next line, etc.
    confidence: float = 0.0
    raw_context: str = dc_field(default="", repr=False)


class BaseExtractor:
    """
    Subclasses set, in __init__:
      - self.anchors: Dict[field_name, List[keyword_regex_str]]
      - self.value_parsers: Dict[field_name, Callable[[str], List[str]]]
    then call self._compile_anchors().
    """

    WINDOW_CHARS = 100
    MAX_LOOKAHEAD_LINES = 2

    # How much leading junk (bullets, whitespace, punctuation) before an
    # anchor on its own line is tolerated before we assume the anchor is
    # actually just a keyword buried inside a sentence, not a form label.
    MAX_LABEL_LINE_PREFIX = 3

    def __init__(self):
        self.anchors: Dict[str, List[str]] = {}
        self.value_parsers: Dict[str, Callable[[str], List[str]]] = {}
        self._compiled_anchors: Dict[str, List[re.Pattern]] = {}

    def _compile_anchors(self):
        self._compiled_anchors = {
            field_name: [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]
            for field_name, patterns in self.anchors.items()
        }

    def _get_window(self, text: str, anchor_start: int, anchor_end: int, window_chars: Optional[int] = None) -> str:
        """
        Convenience single-window helper (used by callers, e.g.
        PartyExtractor, that want one merged window rather than the
        try-same-line-then-lookahead sequence _get_windows applies).
        """
        windows = self._get_windows(text, anchor_start, anchor_end, window_chars)
        return windows[-1][0] if windows else ""

    def _get_windows(self, text: str, anchor_start: int, anchor_end: int, window_chars: Optional[int] = None) -> List[tuple]:
        """
        Build the context window(s) that follow an anchor match, as a
        list of (window_text, line_offset) tried in priority order:
          1. The remainder of the current line alone. Tried first and
             on its own, so a short-but-valid same-line value (e.g. a
             book number of just "1") is never passed over.
          2. Only if that line is empty (the anchor sits right at the
             end of its line -- the OCR label/value pair got split
             across lines) AND the anchor itself was near the start of
             its line (i.e. it reads as a form label, not a keyword
             that happens to close out an unrelated sentence), a short
             lookahead into the next 1-2 non-empty lines.
        Requiring the anchor to be line-initial before allowing
        lookahead is what stops a phrase like "...the total
        consideration" (mid-sentence) from grabbing an unrelated value
        off the next line just because it happens to fall at a line
        break.
        """
        limit = window_chars or self.WINDOW_CHARS
        remainder = text[anchor_end:anchor_end + limit * 3]
        lines = remainder.split("\n")
        windows = []

        first_line = lines[0].strip()
        if first_line:
            windows.append((first_line[:limit], 0))
        else:
            line_start = text.rfind("\n", 0, anchor_start) + 1
            prefix = text[line_start:anchor_start].strip()
            anchor_is_label_like = len(prefix) <= self.MAX_LABEL_LINE_PREFIX
            if anchor_is_label_like:
                for l in lines[1:1 + self.MAX_LOOKAHEAD_LINES]:
                    l = l.strip()
                    if l:
                        windows.append((l[:limit], 1))
                        break

        return windows

    def _find_candidates(self, text: str, field_name: str) -> List[FieldCandidate]:
        candidates = []
        patterns = self._compiled_anchors.get(field_name, [])
        parser = self.value_parsers.get(field_name)
        if not parser:
            return candidates

        for pattern in patterns:
            for m in pattern.finditer(text):
                for window, line_offset in self._get_windows(text, m.start(), m.end()):
                    values = parser(window)
                    if not values:
                        continue
                    for v in values:
                        if not v:
                            continue
                        pos = window.find(v)
                        candidates.append(FieldCandidate(
                            field_name=field_name,
                            value=v,
                            anchor_keyword=m.group(0),
                            distance=pos if pos >= 0 else len(window),
                            line_offset=line_offset,
                            raw_context=window,
                        ))
                    break  # same-line window already yielded a value; skip lookahead
        return candidates

    def _score_candidate(self, candidate: FieldCandidate, validator: Optional[Callable[[str, str], bool]] = None) -> float:
        if validator and not validator(candidate.field_name, candidate.value):
            return 0.0
        score = 70.0
        score -= min(candidate.distance, 40) * 0.5
        score -= candidate.line_offset * 10
        if len(candidate.value) < 2:
            score -= 20
        return max(0.0, min(100.0, score))

    def _select_best(self, candidates: List[FieldCandidate], validator: Optional[Callable[[str, str], bool]] = None) -> Optional[FieldCandidate]:
        if not candidates:
            return None
        for c in candidates:
            c.confidence = self._score_candidate(c, validator)
        valid = [c for c in candidates if c.confidence > 0]
        if not valid:
            return None
        valid.sort(key=lambda c: c.confidence, reverse=True)
        return valid[0]

    @staticmethod
    def clean_value(value: str) -> str:
        if not value:
            return ""
        value = value.strip()
        value = re.sub(r"[;:,\.]+$", "", value)
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    @staticmethod
    def dedupe(values: List[str]) -> List[str]:
        seen = set()
        unique = []
        for v in values:
            key = re.sub(r"\s+", "", v).lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(v)
        return unique


# ============================================================================
# DocumentExtractor -- deed/registration metadata + document type
# ============================================================================

class DocumentExtractor(BaseExtractor):

    FIELDS = [
        "deed_number", "token_number", "registration_number", "serial_number",
        "registration_date", "registration_office", "book_number",
        "volume_number", "page_number",
    ]

    DOCUMENT_TYPE_KEYWORDS = {
        "sale_deed": [r"\bSale\s+Deed\b", r"à¤¬à¥ˆà¤¨à¤¾à¤®à¤¾", r"à¤¬à¤¿à¤•à¥à¤°à¥€\s+à¤ªà¤¤à¥à¤°"],
        "gift_deed": [r"\bGift\s+Deed\b", r"à¤¦à¤¾à¤¨\s+à¤ªà¤¤à¥à¤°"],
        "mortgage_deed": [r"\bMortgage\s+Deed\b", r"à¤¬à¤‚à¤§à¤•\s+à¤ªà¤¤à¥à¤°"],
        "lease_deed": [r"\bLease\s+Deed\b", r"à¤ªà¤Ÿà¥à¤Ÿà¤¾(?:\s+à¤ªà¤¤à¥à¤°)?"],
        "partition_deed": [r"\bPartition\s+Deed\b", r"à¤¬à¤‚à¤Ÿà¤µà¤¾à¤°à¤¾\s+à¤ªà¤¤à¥à¤°"],
        "will": [r"\bWill\b", r"à¤µà¤¸à¥€à¤¯à¤¤"],
    }

    def __init__(self):
        super().__init__()
        self.anchors = {
            "deed_number": [r"\bDeed\s*No\.?", r"\bDocument\s*No\.?", r"à¤¦à¤¸à¥à¤¤à¤¾à¤µà¥‡à¤œà¤¼\s*à¤¸à¤‚à¤–à¥à¤¯à¤¾"],
            "token_number": [r"\bToken\s*No\.?", r"\bToken\s*Number\b", r"à¤Ÿà¥‹à¤•à¤¨\s*à¤¨à¤‚\.?"],
            "registration_number": [
                r"\bRegistration\s*Number\b", r"\bReg\.?\s*No\.?",
                r"à¤ªà¤‚à¤œà¥€à¤•à¤°à¤£\s*à¤¸à¤‚à¤–à¥à¤¯à¤¾", r"à¤ªà¤‚\.\s*à¤¸à¤‚\.?",
            ],
            "serial_number": [
                r"\bSerial\s*Number\b", r"\bSerial\s*No\.?", r"\bS\.No\.?",
                r"à¤•à¥à¤°à¤®\s*à¤¸à¤‚à¤–à¥à¤¯à¤¾",
            ],
            "registration_date": [
                r"\bRegistration\s*Date\b", r"\bDate\s*of\s*Registration\b",
                r"à¤¦à¤¿à¤¨à¤¾à¤‚à¤•", r"\bDate\b",
            ],
            "registration_office": [
                r"\bRegistration\s*Office\b", r"\bOffice\s*of\s*the\s*Sub\s*Registrar\b",
                r"à¤ªà¤‚à¤œà¥€à¤•à¤°à¤£\s*à¤•à¤¾à¤°à¥à¤¯à¤¾à¤²à¤¯",
            ],
            "book_number": [r"\bBook\s*No\.?", r"\bBook\s*Number\b", r"à¤¬à¥à¤•\s*à¤¨à¤‚\.?"],
            "volume_number": [r"\bVolume\s*No\.?", r"\bVolume\s*Number\b", r"à¤µà¥‰à¤²à¥à¤¯à¥‚à¤®\s*à¤¨à¤‚\.?"],
            "page_number": [r"\bPage\s*No\.?", r"\bPage\s*Number\b", r"à¤ªà¥ƒà¤·à¥à¤ \s*à¤¸à¤‚à¤–à¥à¤¯à¤¾"],
        }
        self.value_parsers = {
            "deed_number": parse_alnum_code,
            "token_number": parse_alnum_code,
            "registration_number": parse_alnum_code,
            "serial_number": parse_alnum_code,
            "registration_date": parse_date,
            "registration_office": parse_place_name,
            "book_number": parse_alnum_code,
            "volume_number": parse_alnum_code,
            "page_number": parse_alnum_code,
        }
        self._compile_anchors()

    def extract(self, text: str) -> dict:
        result = {f: "" for f in self.FIELDS}
        result["document_type"] = ""

        filled = 0
        for field_name in self.FIELDS:
            candidates = self._find_candidates(text, field_name)
            best = self._select_best(candidates, validator=self._validate)
            if best:
                result[field_name] = self.clean_value(best.value)
                filled += 1

        doc_type = self._classify_document_type(text)
        result["document_type"] = doc_type
        if doc_type:
            filled += 1

        total_fields = len(self.FIELDS) + 1  # +1 for document_type
        result["_confidence"] = round((filled / total_fields) * 100, 1)
        return result

    def _classify_document_type(self, text: str) -> str:
        for doc_type, patterns in self.DOCUMENT_TYPE_KEYWORDS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return doc_type
        return ""

    @staticmethod
    def _validate(field_name: str, value: str) -> bool:
        return bool(value) and len(value) >= 1


# ============================================================================
# FinancialExtractor -- monetary fields
# ============================================================================

class FinancialExtractor(BaseExtractor):

    FIELDS = ["stamp_duty", "registration_fee", "sale_consideration", "market_value", "other_fee"]

    def __init__(self):
        super().__init__()
        self.anchors = {
            "sale_consideration": [
                r"\bSale\s+Consideration\b", r"\bConsideration\s+Amount\b",
                r"\bTotal\s+Consideration\b", r"à¤¬à¤¿à¤•à¥à¤°à¥€\s+à¤®à¥‚à¤²à¥à¤¯", r"à¤•à¥à¤²\s+à¤°à¤¾à¤¶à¤¿",
            ],
            "market_value": [
                r"\bMarket\s+Value\b", r"\bGuidance\s+Value\b", r"\bCircle\s+Rate\b",
                r"à¤®à¤¾à¤°à¥à¤•à¥‡à¤Ÿ\s+à¤µà¥ˆà¤²à¥à¤¯à¥‚", r"à¤®à¤¾à¤°à¥à¤—à¤¦à¤°à¥à¤¶à¤•\s+à¤®à¥‚à¤²à¥à¤¯", r"à¤¸à¤°à¥à¤•à¤¿à¤²\s+à¤°à¥‡à¤Ÿ",
            ],
            "registration_fee": [
                r"\bRegistration\s+Fee\b", r"\bReg\.?\s*Fee\b",
                r"à¤ªà¤‚à¤œà¥€à¤•à¤°à¤£\s+à¤¶à¥à¤²à¥à¤•", r"à¤°à¤œà¤¿à¤¸à¥à¤Ÿà¥à¤°à¥‡à¤¶à¤¨\s+à¤«à¥€à¤¸",
            ],
            "stamp_duty": [r"\bStamp\s+Duty\b", r"à¤¸à¥à¤Ÿà¤¾à¤®à¥à¤ª\s+à¤¡à¥à¤¯à¥‚à¤Ÿà¥€", r"à¤¸à¥à¤Ÿà¤¾à¤®à¥à¤ª\s+à¤¶à¥à¤²à¥à¤•"],
            "other_fee": [r"\bOther\s+Fee\b", r"\bMiscellaneous\s+Fee\b", r"\bAdditional\s+Fee\b"],
        }
        self.value_parsers = {f: parse_currency for f in self.FIELDS}
        self._compile_anchors()

    def extract(self, text: str) -> dict:
        result = {f: "" for f in self.FIELDS}
        filled = 0
        for field_name in self.FIELDS:
            candidates = self._find_candidates(text, field_name)
            best = self._select_best(candidates, validator=self._validate_amount)
            if best:
                result[field_name] = self.clean_value(best.value)
                filled += 1
        result["_confidence"] = round((filled / len(self.FIELDS)) * 100, 1)
        return result

    def extract_sale_consideration_candidates(self, text: str) -> list:
        """
        Backward-compatible with the original RegexExtractor method:
        returns every plausible sale-consideration mention, deduplicated
        and ranked by numeric value (highest first), for pipelines that
        cross-check the figure against multiple places it's quoted.
        """
        candidates = self._find_candidates(text, "sale_consideration")
        out = []
        seen = set()
        for c in candidates:
            value = self.clean_value(c.value)
            if not value or not self._validate_amount("sale_consideration", value):
                continue
            key = re.sub(r"[^\d.]", "", value)
            if not key or key in seen:
                continue
            seen.add(key)
            try:
                normalized = float(value.replace(",", ""))
            except ValueError:
                normalized = None
            out.append({"raw": value, "normalized": normalized})
        out.sort(key=lambda x: x["normalized"] or 0, reverse=True)
        return out

    @staticmethod
    def _validate_amount(field_name: str, value: str) -> bool:
        digits = re.sub(r"[^0-9]", "", value)
        if len(digits) < 3:
            return False
        total = len(value.replace(" ", "").replace(".", "").replace(",", ""))
        if total == 0:
            return False
        ascii_count = sum(1 for c in value if ord(c) < 128)
        return (ascii_count / total) >= 0.4


# ============================================================================
# PropertyExtractor -- location and parcel-identifier fields
# ============================================================================

class PropertyExtractor(BaseExtractor):

    FIELDS = [
        "district", "village", "survey_number", "plot_number",
        "khasra_number", "khata_number", "area", "tehsil",
    ]

    def __init__(self):
        super().__init__()
        self.anchors = {
            "district": [r"\bDistrict\b", r"à¤œà¤¿à¤²à¤¾"],
            "village": [r"\bVillage\b", r"à¤—à¤¾à¤à¤µ", r"à¤—à¤¾à¤‚à¤µ"],
            "survey_number": [r"\bSurvey\s*No\.?", r"\bSurvey\s*Number\b", r"à¤¸à¤°à¥à¤µà¥‡\s*à¤¸à¤‚à¤–à¥à¤¯à¤¾", r"à¤¸à¤°à¥à¤µà¥‡\s*à¤¨à¤‚\.?"],
            "plot_number": [r"\bPlot\s*No\.?", r"\bPlot\s*Number\b", r"à¤ªà¥à¤²à¥‰à¤Ÿ\s*à¤¨à¤‚\.?", r"à¤ªà¥à¤²à¥‰à¤Ÿ\s*à¤¸à¤‚à¤–à¥à¤¯à¤¾"],
            "khasra_number": [r"\bKhasra\s*No\.?", r"\bKhasra\s*Number\b", r"à¤–à¤¸à¤°à¤¾\s*à¤¨à¤‚\.?", r"à¤–à¤¸à¤°à¤¾\s*à¤¸à¤‚à¤–à¥à¤¯à¤¾"],
            "khata_number": [
                r"\bKhata\s*No\.?", r"\bKhata\s*Number\b", r"\bKhatian\s*No\.?",
                r"à¤–à¤¾à¤¤à¤¾\s*à¤¨à¤‚\.?", r"à¤–à¤¾à¤¤à¤¾\s*à¤¸à¤‚à¤–à¥à¤¯à¤¾",
            ],
            "area": [r"\bArea\b", r"\bLand\s*Area\b", r"à¤•à¥à¤·à¥‡à¤¤à¥à¤°à¤«à¤²", r"à¤•à¥à¤·à¥‡à¤¤à¥à¤°"],
            "tehsil": [r"\bTehsil\b", r"\bTaluka\b", r"à¤¤à¤¹à¤¸à¥€à¤²", r"à¤¤à¤¾à¤²à¥à¤•à¤¾"],
        }
        self.value_parsers = {
            "district": parse_place_name,
            "village": parse_place_name,
            "survey_number": parse_alnum_code,
            "plot_number": parse_alnum_code,
            "khasra_number": parse_alnum_code,
            "khata_number": parse_alnum_code,
            "area": parse_area,
            "tehsil": parse_place_name,
        }
        self._compile_anchors()

    def extract(self, text: str) -> dict:
        result = {f: "" for f in self.FIELDS}
        filled = 0
        for field_name in self.FIELDS:
            candidates = self._find_candidates(text, field_name)
            best = self._select_best(candidates, validator=self._validate)
            if best:
                result[field_name] = self.clean_value(best.value)
                filled += 1
        result["_confidence"] = round((filled / len(self.FIELDS)) * 100, 1)
        return result

    @staticmethod
    def _validate(field_name: str, value: str) -> bool:
        return bool(value) and len(value.strip()) >= 2


# ============================================================================
# PartyExtractor -- seller/buyer/witness name, relation, address
# ============================================================================

class PartyExtractor(BaseExtractor):
    """
    New extractor (no equivalent existed in the original RegexExtractor):
    pulls out the people named on the deed -- first party (seller/vendor),
    second party (buyer/purchaser/vendee), and witnesses -- along with
    whatever relation ("S/O"/"D/O"/"W/O") and address text sits next to
    their name.

    Party blocks are structurally different from single-value fields (one
    anchor can yield a *group* of related values: name, relation, address),
    so this extractor builds its own per-anchor parser instead of reusing
    BaseExtractor's single-value _find_candidates/_select_best pipeline --
    it still reuses _get_window and clean_value from the base class.
    """

    ROLE_KEYWORDS = {
        "seller": [r"\bSeller(?:s)?\b", r"\bVendor(?:s)?\b", r"\bFirst\s+Party\b", r"à¤µà¤¿à¤•à¥à¤°à¥‡à¤¤à¤¾", r"à¤ªà¥à¤°à¤¥à¤®\s+à¤ªà¤•à¥à¤·"],
        "buyer": [r"\bPurchaser(?:s)?\b", r"\bBuyer(?:s)?\b", r"\bVendee(?:s)?\b", r"\bSecond\s+Party\b", r"à¤•à¥à¤°à¥‡à¤¤à¤¾", r"à¤¦à¥à¤µà¤¿à¤¤à¥€à¤¯\s+à¤ªà¤•à¥à¤·"],
        "witness": [r"\bWitness(?:es)?\b", r"à¤—à¤µà¤¾à¤¹"],
    }

    ROLE_TO_KEY = {"seller": "first_party", "buyer": "second_party", "witness": "witnesses"}

    RELATION_PATTERNS = [
        (r"(?:S/O|S\.O\.|Son\s+of|à¤ªà¥à¤¤à¥à¤°)", "son_of"),
        (r"(?:D/O|D\.O\.|Daughter\s+of|à¤ªà¥à¤¤à¥à¤°à¥€)", "daughter_of"),
        (r"(?:W/O|W\.O\.|Wife\s+of|à¤ªà¤¤à¥à¤¨à¥€)", "wife_of"),
    ]

    NAME_STOP = (
        r"(?:,|\n|S/O|D/O|W/O|R/O|Son\s+of|Daughter\s+of|Wife\s+of|"
        r"resident\s+of|aged|à¤ªà¥à¤¤à¥à¤°|à¤ªà¥à¤¤à¥à¤°à¥€|à¤ªà¤¤à¥à¤¨à¥€|à¤†à¤¯à¥|à¤¨à¤¿à¤µà¤¾à¤¸à¥€|$)"
    )

    WINDOW_CHARS_PARTY = 180

    def __init__(self):
        super().__init__()
        self.anchors = self.ROLE_KEYWORDS
        self._compile_anchors()

    def extract(self, text: str) -> dict:
        result = {"first_party": [], "second_party": [], "witnesses": []}

        for role, target_key in self.ROLE_TO_KEY.items():
            for pattern in self._compiled_anchors.get(role, []):
                for m in pattern.finditer(text):
                    window = self._get_window(text, m.start(), m.end(), window_chars=self.WINDOW_CHARS_PARTY)
                    party = self._parse_party_block(window)
                    if party:
                        party["role"] = role
                        result[target_key].append(party)

        for key in ("first_party", "second_party", "witnesses"):
            result[key] = self._dedupe_parties(result[key])

        filled = sum(1 for k in ("first_party", "second_party", "witnesses") if result[k])
        result["_confidence"] = round((filled / 3) * 100, 1)
        return result

    def _parse_party_block(self, window: str) -> dict:
        name = ""
        name_match = re.search(
            r"[:\-â€“â€”]?\s*([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\.\s]{1,50}?)" + self.NAME_STOP,
            window, re.IGNORECASE,
        )
        if name_match:
            name = name_match.group(1).strip()
        if not name:
            return {}

        relation_type, relation_name = "", ""
        for pattern, rel_type in self.RELATION_PATTERNS:
            rm = re.search(
                pattern + r"[\s:\-]*([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\.\s]{1,40}?)"
                r"(?:,|\n|R/O|resident|aged|à¤¨à¤¿à¤µà¤¾à¤¸à¥€|à¤†à¤¯à¥|$)",
                window, re.IGNORECASE,
            )
            if rm:
                relation_type = rel_type
                relation_name = rm.group(1).strip()
                break

        address = ""
        address_match = re.search(
            r"(?:R/O|resident\s+of|à¤¨à¤¿à¤µà¤¾à¤¸à¥€)[\s:\-]*([A-Za-z\u0900-\u097F0-9,\.\s]{3,80})",
            window, re.IGNORECASE,
        )
        if address_match:
            address = address_match.group(1).strip()

        return {
            "name": self.clean_value(name),
            "relation_type": relation_type,
            "relation_name": self.clean_value(relation_name),
            "address": self.clean_value(address),
        }

    @staticmethod
    def _dedupe_parties(parties: list) -> list:
        seen = set()
        unique = []
        for p in parties:
            key = re.sub(r"\s+", "", p["name"]).lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(p)
        return unique


# ============================================================================
# ExtractionEngine -- orchestrates all four extractors
# ============================================================================

class ExtractionEngine:
    """
    engine = ExtractionEngine()
    result = engine.extract(ocr_text)

    result == {
        "document_details": {..., "_confidence": float},
        "financial":        {..., "_confidence": float},
        "property":         {..., "_confidence": float},
        "parties":          {"first_party": [...], "second_party": [...],
                              "witnesses": [...], "_confidence": float},
    }

    Drop-in compatible with the legacy RegexExtractor's output shape
    (document_details / financial / property, each with "_confidence"),
    plus the new "parties" section.
    """

    def __init__(self):
        self.document_extractor = DocumentExtractor()
        self.financial_extractor = FinancialExtractor()
        self.property_extractor = PropertyExtractor()
        self.party_extractor = PartyExtractor()

    def extract(self, text: str) -> dict:
        if not text or not text.strip():
            return self._empty_result()
        return {
            "document_details": self.document_extractor.extract(text),
            "financial": self.financial_extractor.extract(text),
            "property": self.property_extractor.extract(text),
            "parties": self.party_extractor.extract(text),
        }

    def extract_sale_consideration_candidates(self, text: str) -> list:
        """Preserved for callers that relied on the legacy RegexExtractor API."""
        return self.financial_extractor.extract_sale_consideration_candidates(text)

    def extract_field_group(self, text: str, group: str) -> dict:
        """
        Run a single extractor by name: 'document', 'financial',
        'property', or 'parties'. Useful when a caller only needs to
        (re)run one section -- e.g. an ExtractionAgent that already got
        a good LLM read on everything except the parties block.
        """
        mapping = {
            "document": self.document_extractor,
            "financial": self.financial_extractor,
            "property": self.property_extractor,
            "parties": self.party_extractor,
        }
        extractor = mapping.get(group)
        if extractor is None:
            raise ValueError(f"Unknown extraction group: {group!r}. "
                              f"Expected one of {list(mapping)}.")
        return extractor.extract(text)

    @staticmethod
    def _empty_result() -> dict:
        return {
            "document_details": {"_confidence": 0},
            "financial": {"_confidence": 0},
            "property": {"_confidence": 0},
            "parties": {"first_party": [], "second_party": [], "witnesses": [], "_confidence": 0},
        }


# ============================================================================
# ExtractionAgentAdapter -- backfill-only integration point for an
# existing ExtractionAgent that runs an LLM pass first
# ============================================================================

class ExtractionAgentAdapter:
    """
    Thin integration layer for an ExtractionAgent that runs an LLM pass
    first and wants this engine only to backfill whatever the LLM left
    blank -- mirroring the original RegexExtractor's stated role
    ("falls back to regex patterns for fields the 3B LLM misses").

    Wiring it into an existing ExtractionAgent:

        class ExtractionAgent:
            def __init__(self):
                self.llm = ...
                self.fallback = ExtractionAgentAdapter()

            def extract(self, ocr_text: str) -> dict:
                llm_result = self.llm.extract(ocr_text)
                return self.fallback.fill_missing(ocr_text, llm_result)

    `llm_result` is expected to already follow the same section/field
    shape as ExtractionEngine.extract() (document_details / financial /
    property / parties). Any field the LLM left empty is backfilled;
    the LLM's own values are never overwritten. Each section's
    "_confidence" is recomputed from the fallback pass since it
    reflects how much of that section came from context matching vs.
    the (presumably higher-trust) LLM.
    """

    def __init__(self):
        self.engine = ExtractionEngine()

    def fill_missing(self, text: str, llm_result: dict) -> dict:
        fallback_result = self.engine.extract(text)
        merged = {}
        for section, fallback_fields in fallback_result.items():
            llm_fields = llm_result.get(section, {}) if llm_result else {}

            if section == "parties":
                merged[section] = self._merge_parties(llm_fields, fallback_fields)
                continue

            merged_section = dict(llm_fields) if isinstance(llm_fields, dict) else {}
            for field_name, fallback_value in fallback_fields.items():
                if field_name == "_confidence":
                    continue
                if not merged_section.get(field_name):
                    merged_section[field_name] = fallback_value
            merged_section["_confidence"] = fallback_fields.get("_confidence", 0)
            merged[section] = merged_section
        return merged

    @staticmethod
    def _merge_parties(llm_parties: dict, fallback_parties: dict) -> dict:
        llm_parties = llm_parties if isinstance(llm_parties, dict) else {}
        merged = {}
        for key in ("first_party", "second_party", "witnesses"):
            llm_list = llm_parties.get(key) or []
            merged[key] = llm_list if llm_list else fallback_parties.get(key, [])
        merged["_confidence"] = fallback_parties.get("_confidence", 0)
        return merged
