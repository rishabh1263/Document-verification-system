"""
================================================================================
ExtractionEngine â€” Production-Ready Modular OCR Extraction System
v2.1.1  (Fixed: block termination, label collisions, Hindi body-word false 
         positives, concatenated labels, name validation, regex over-capture)
================================================================================
"""

import re
import json
import unicodedata
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ExtractedField:
    value: Any
    confidence: float = 0.0
    source: str = ""
    method: str = ""
    raw_matches: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "method": self.method,
            "raw_matches": self.raw_matches,
        }


class BaseExtractor:
    NAME = "base"
    MIN_NAME_LENGTH = 3
    MAX_NAME_LENGTH = 60
    MAX_NAME_WORDS = 8

    # Lines containing these words terminate the current block (prevents runaway collection)
    BLOCK_STOP_WORDS = {
        "signature", "photo", "thumb", "index", "ring", "self", "scanned",
        "endorsement", "registered", "registering", "fee", "total",
        "document", "instrument", "pages of the", "==========",
    }

    GARBAGE_WORDS = {
        "photo", "thumb", "signature", "index", "ring", "self",
        "thumb impression", "photo id", "signature of", "sign",
        "page", "no", "number", "date", "deed", "registry", "serial",
        "stamp", "duty", "amount", "rs", "rupees", "address",
        "son", "s/o", "d/o", "w/o", "c/o", "age", "occupation",
        "resident", "r/o", "do", "to", "from", "by", "and", "or",
        "status", "profession", "middle", "presentant", "presenter",
        "executant", "identifier", "witness", "claimant", "party",
        "scanned", "oken", "scanner", "score", "ver",
        "à¤«à¥‹à¤Ÿà¥‹", "à¤…à¤‚à¤—à¥‚à¤ à¤¾", "à¤¹à¤¸à¥à¤¤à¤¾à¤•à¥à¤·à¤°", "à¤…à¤¨à¥à¤•à¥à¤°à¤®à¤£à¤¿à¤•à¤¾", "à¤…à¤‚à¤—à¥‚à¤ à¥€", "à¤¸à¥à¤µà¤¯à¤‚",
        "à¤ªà¥ƒà¤·à¥à¤ ", "à¤•à¥à¤°à¤®à¤¾à¤‚à¤•", "à¤¤à¤¾à¤°à¥€à¤–", "à¤°à¤œà¤¿à¤¸à¥à¤Ÿà¥à¤°à¥€", "à¤¸à¥€à¤°à¤¿à¤¯à¤²",
        "à¤¸à¥à¤Ÿà¤¾à¤®à¥à¤ª", "à¤¡à¥à¤¯à¥‚à¤Ÿà¥€", "à¤°à¤¾à¤¶à¤¿", "à¤ªà¤¤à¤¾", "à¤ªà¥à¤¤à¥à¤°", "à¤ªà¤¤à¤¿", "à¤ªà¤¤à¥à¤¨à¥€",
        "à¤‰à¤®à¥à¤°", "à¤ªà¥‡à¤¶à¤¾", "à¤¨à¤¿à¤µà¤¾à¤¸à¥€", "à¤®à¤§à¥à¤¯", "à¤…à¤¨à¥à¤ªà¤¸à¥à¤¥à¤¿à¤¤", "à¤¹à¤¸à¥à¤¤à¤¾à¤•à¥à¤·à¤°à¤¿à¤¤",
        "à¤µà¤¿à¤•à¥à¤°à¥‡à¤¤à¤¾", "à¤•à¥à¤°à¥‡à¤¤à¤¾", "à¤µà¤¿à¤•à¥à¤°à¤¯", "à¤µà¤¿à¤²à¥‡à¤–", "à¤¸à¤‚à¤ªà¤¤à¥à¤¤à¤¿", "à¤ªà¥à¤°à¤¸à¥à¤¤à¤¾à¤µ",
        "à¤¸à¥à¤µà¥€à¤•à¤¾à¤°", "à¤¹à¤¸à¥à¤¤à¤¾à¤‚à¤¤à¤°à¤£", "à¤¸à¤®à¤°à¥à¤ªà¤£", "à¤¸à¤¾à¤•à¥à¤·à¥€", "à¤¸à¤¾à¤•à¥à¤·", "à¤˜à¥‹à¤·à¤¿à¤¤",
        "à¤¨à¤¿à¤°à¥€à¤•à¥à¤·à¤£", "à¤ªà¤°à¥€à¤•à¥à¤·à¤£", "à¤¦à¤¸à¥à¤¤à¤¾à¤µà¥‡à¤œà¤¼", "à¤¦à¤¸à¥à¤¤à¤¾à¤µà¥‡à¤œ", "à¤­à¥à¤—à¤¤à¤¾à¤¨",
        "à¤¸à¥à¤ªà¥à¤°à¥à¤¦", "à¤¨à¤¿à¤µà¤‚à¤§à¤¨", "à¤ªà¤‚à¤œà¥€à¤•à¤°à¤£", "à¤•à¤¾à¤°à¥à¤¯à¤¾à¤²à¤¯", "à¤œà¤¿à¤²à¤¾", "à¤¥à¤¾à¤¨à¤¾",
        "à¤…à¤šà¤²", "à¤®à¤•à¤¾à¤¨", "à¤­à¥‚à¤®à¤¿", "à¤œà¤®à¥€à¤¨", "à¤ªà¥à¤²à¥‰à¤Ÿ", "à¤–à¤¸à¤°à¤¾", "à¤–à¤¾à¤¤à¤¾",
        "à¤¸à¤°à¥à¤µà¥‡", "à¤•à¥à¤·à¥‡à¤¤à¥à¤°à¤«à¤²", "à¤°à¤•à¤¬à¤¾", "à¤¬à¤•à¤¾à¤¯à¤¾", "à¤¬à¤¿à¤œà¤²à¥€", "à¤•à¤°",
        "à¤®à¥à¤•à¤¦à¤®à¤¾", "à¤¨à¥à¤¯à¤¾à¤¯à¤¾à¤²à¤¯à¥€à¤¨", "à¤µà¤¿à¤µà¤¾à¤¦", "à¤¦à¤¾à¤¯à¤¿à¤¤à¥à¤µ", "à¤¶à¥à¤²à¥à¤•",
        "à¤®à¥‚à¤²à¥à¤¯", "à¤°à¥à¤ªà¤¯à¥‡", "à¤°à¥à¤ªà¤¯à¤¾", "à¤°à¥‚à¤ªà¤¯à¥‡", "à¤¸à¤‚à¤¤à¥‹à¤·", "à¤¸à¥à¤µà¤¾à¤®à¥€",
        "à¤µà¤°à¥à¤£à¤¿à¤¤", "à¤‰à¤ªà¤°à¥‹à¤•à¥à¤¤", "à¤…à¤§à¤¿à¤•à¤¾à¤°", "à¤ªà¥à¤°à¤¥à¤®", "à¤¸à¤¾à¤®à¥‚à¤¹à¤¿à¤•",
        "à¤•à¤¹à¤¾", "à¤œà¤¾à¤à¤—à¤¾", "à¤œà¤¬à¤•à¤¿", "à¤¦à¥à¤µà¤¾à¤°à¤¾", "à¤¹à¥‡à¤¤à¥", "à¤¤à¤¥à¤¾", "à¤à¤µà¤‚",
        "à¤…à¤¥à¤µà¤¾", "à¤¯à¤¦à¤¿", "à¤•à¤¿", "à¤•à¤¾", "à¤•à¥‡", "à¤•à¥‹", "à¤¨à¥‡", "à¤¸à¥‡",
        "à¤®à¥‡à¤‚", "à¤ªà¤°", "à¤¹à¥ˆ", "à¤¹à¥ˆà¤‚", "à¤¥à¤¾", "à¤¥à¥€", "à¤¥à¥‡", "à¤¹à¥‹à¤‚à¤—à¥‡",
        "à¤•à¤°", "à¤•à¤°à¤¤à¤¾", "à¤•à¤°à¤¤à¥‡", "à¤•à¤¿à¤¯à¤¾", "à¤—à¤¯à¤¾", "à¤—à¤ˆ", "à¤—à¤",
        "à¤²à¤¿à¤¯à¤¾", "à¤¦à¤¿à¤¯à¤¾", "à¤ªà¤¾à¤¯à¤¾", "à¤®à¤¿à¤²à¤¾", "à¤¹à¥à¤†", "à¤œà¤¾à¤¤à¤¾", "à¤œà¤¾à¤¤à¥€",
        "à¤œà¤¾à¤¤à¥‡", "à¤°à¤¹à¤¾", "à¤°à¤¹à¥€", "à¤°à¤¹à¥‡", "à¤¸à¤•à¤¤à¤¾", "à¤¸à¤•à¤¤à¥€", "à¤¸à¤•à¤¤à¥‡",
        "à¤šà¤¾à¤¹à¤¿à¤", "à¤ªà¤¡à¤¼à¤¤à¤¾", "à¤ªà¤¡à¤¼à¤¤à¥€", "à¤ªà¤¡à¤¼à¤¤à¥‡", "à¤µà¤¾à¤²à¤¾", "à¤µà¤¾à¤²à¥€", "à¤µà¤¾à¤²à¥‡",
        "à¤•à¤°à¤¨à¥‡", "à¤¹à¥‹à¤¨à¥‡", "à¤¦à¥‡à¤¨à¥‡", "à¤²à¥‡à¤¨à¥‡", "à¤†à¤¨à¥‡", "à¤œà¤¾à¤¨à¥‡",
        "à¤†à¤¦à¤¿", "à¤‡à¤¤à¥à¤¯à¤¾à¤¦à¤¿", "à¤†à¤¦à¥‡à¤¶", "à¤†à¤¦à¥‡à¤¶à¤¾à¤¨à¥à¤¸à¤¾à¤°", "à¤…à¤¨à¥à¤¸à¤¾à¤°",
        "à¤…à¤¤à¤¿à¤°à¤¿à¤•à¥à¤¤", "à¤…à¤¨à¥à¤¯", "à¤…à¤¨à¥à¤¯à¤¥à¤¾", "à¤…à¤°à¥à¤¥à¤¾à¤¤à¥", "à¤…à¤°à¥à¤¥à¤¾à¤¤",
        "à¤‡à¤¸", "à¤‡à¤¸à¥€", "à¤‡à¤¸à¤•à¤¾", "à¤‡à¤¸à¤•à¥€", "à¤‡à¤¸à¤•à¥‡", "à¤‡à¤¸à¤•à¥‹", "à¤‡à¤¸à¤¨à¥‡",
        "à¤‰à¤¸", "à¤‰à¤¸à¥€", "à¤‰à¤¸à¤•à¤¾", "à¤‰à¤¸à¤•à¥€", "à¤‰à¤¸à¤•à¥‡", "à¤‰à¤¸à¤•à¥‹", "à¤‰à¤¸à¤¨à¥‡",
        "à¤œà¤¿à¤¸", "à¤œà¤¿à¤¸à¤•à¤¾", "à¤œà¤¿à¤¸à¤•à¥€", "à¤œà¤¿à¤¸à¤•à¥‡", "à¤œà¤¿à¤¸à¤•à¥‹", "à¤œà¤¿à¤¸à¤¨à¥‡",
        "à¤¸à¤¬", "à¤¸à¤­à¥€", "à¤¸à¤¬à¤•à¤¾", "à¤¸à¤¬à¤•à¥€", "à¤¸à¤¬à¤•à¥‡", "à¤¸à¤¬à¤•à¥‹", "à¤¸à¤¬à¤¨à¥‡",
        "à¤•à¥‹à¤ˆ", "à¤•à¤¿à¤¸à¥€", "à¤•à¥à¤›", "à¤•à¤¹à¥€à¤‚", "à¤•à¤­à¥€", "à¤•à¥ˆà¤¸à¥‡", "à¤•à¥à¤¯à¥‹à¤‚",
        "à¤•à¥à¤¯à¥‹à¤‚à¤•à¤¿", "à¤šà¥‚à¤à¤•à¤¿", "à¤¯à¤¦à¥à¤¯à¤ªà¤¿", "à¤…à¤¤à¤ƒ", "à¤‡à¤¸à¤²à¤¿à¤",
        "à¤¤à¤¾à¤•à¤¿", "à¤œà¥ˆà¤¸à¥‡", "à¤œà¥ˆà¤¸à¤¾", "à¤œà¥ˆà¤¸à¥€", "à¤œà¤¿à¤¤à¤¨à¤¾", "à¤œà¤¿à¤¤à¤¨à¥€", "à¤œà¤¿à¤¤à¤¨à¥‡",
        "à¤µà¥ˆà¤¸à¥‡", "à¤µà¥ˆà¤¸à¤¾", "à¤µà¥ˆà¤¸à¥€", "à¤‰à¤¤à¤¨à¤¾", "à¤‰à¤¤à¤¨à¥€", "à¤‰à¤¤à¤¨à¥‡",
        "à¤­à¥€", "à¤¹à¥€", "à¤¤à¥‹", "à¤¤à¤•", "à¤”à¤°", "à¤¯à¤¾", "à¤²à¥‡à¤•à¤¿à¤¨", "à¤ªà¤°à¤‚à¤¤à¥",
        "à¤•à¤¿à¤‚à¤¤à¥", "à¤«à¤¿à¤°", "à¤¤à¤¬", "à¤…à¤¬", "à¤†à¤œ", "à¤•à¤²", "à¤ªà¤°à¤¸à¥‹à¤‚",
        "à¤¯à¤¹à¤¾à¤", "à¤µà¤¹à¤¾à¤", "à¤•à¤¹à¤¾à¤", "à¤‡à¤§à¤°", "à¤‰à¤§à¤°", "à¤œà¤¿à¤§à¤°", "à¤•à¤¿à¤§à¤°",
        "à¤‡à¤¸à¥€à¤²à¤¿à¤", "à¤‡à¤¸à¤²à¤¿à¤", "à¤œà¤¿à¤¸à¤²à¤¿à¤", "à¤œà¤¿à¤¸à¥€à¤²à¤¿à¤", "à¤¤à¤¥à¤¾à¤ªà¤¿",
        "à¤…à¤ªà¤¨à¤¾", "à¤…à¤ªà¤¨à¥€", "à¤…à¤ªà¤¨à¥‡", "à¤®à¥‡à¤°à¤¾", "à¤®à¥‡à¤°à¥€", "à¤®à¥‡à¤°à¥‡",
        "à¤¤à¥à¤®à¥à¤¹à¤¾à¤°à¤¾", "à¤¤à¥à¤®à¥à¤¹à¤¾à¤°à¥€", "à¤¤à¥à¤®à¥à¤¹à¤¾à¤°à¥‡", "à¤†à¤ªà¤•à¤¾", "à¤†à¤ªà¤•à¥€", "à¤†à¤ªà¤•à¥‡",
        "à¤¹à¤®à¤¾à¤°à¤¾", "à¤¹à¤®à¤¾à¤°à¥€", "à¤¹à¤®à¤¾à¤°à¥‡", "à¤‡à¤¨à¤•à¤¾", "à¤‡à¤¨à¤•à¥€", "à¤‡à¤¨à¤•à¥‡",
        "à¤‰à¤¨à¤•à¤¾", "à¤‰à¤¨à¤•à¥€", "à¤‰à¤¨à¤•à¥‡", "à¤œà¤¿à¤¨à¤•à¤¾", "à¤œà¤¿à¤¨à¤•à¥€", "à¤œà¤¿à¤¨à¤•à¥‡",
        "à¤¸à¥à¤µà¤¯à¤‚", "à¤–à¥à¤¦", "à¤†à¤ª", "à¤®à¥ˆà¤‚", "à¤¹à¤®", "à¤¤à¥à¤®", "à¤¤à¥‚",
        "à¤¯à¤¹", "à¤µà¤¹", "à¤¯à¥‡", "à¤µà¥‡", "à¤‡à¤¨", "à¤‰à¤¨", "à¤œà¤¿à¤¨",
        "à¤œà¥‹", "à¤¸à¥‹", "à¤•à¥à¤¯à¤¾", "à¤•à¥Œà¤¨", "à¤•à¤¿à¤¸", "à¤•à¤¿à¤¨",
        "à¤¨à¤—à¤°", "à¤—à¤¾à¤à¤µ", "à¤—à¤¾à¤‚à¤µ", "à¤®à¥Œà¤œà¤¾", "à¤¤à¤¹à¤¸à¥€à¤²", "à¤¤à¤¾à¤²à¥à¤•à¤¾",
        "à¤œà¤¿à¤²à¥‡", "à¤°à¤¾à¤œà¥à¤¯", "à¤¦à¥‡à¤¶", "à¤¸à¤‚à¤˜", "à¤ªà¥à¤°à¤¦à¥‡à¤¶", "à¤•à¥à¤·à¥‡à¤¤à¥à¤°",
        "à¤®à¤¾à¤°à¥à¤—", "à¤¸à¤¡à¤¼à¤•", "à¤—à¤²à¥€", "à¤®à¥‹à¤¹à¤²à¥à¤²à¤¾", "à¤Ÿà¥‹à¤²à¤¾", "à¤¬à¤¾à¤œà¤¾à¤°",
        "à¤¨à¤¿à¤µà¤¾à¤¸", "à¤¨à¤¿à¤µà¤¾à¤¸ à¤¸à¥à¤¥à¤¾à¤¨", "à¤¸à¥à¤¥à¤¾à¤¯à¥€", "à¤…à¤¸à¥à¤¥à¤¾à¤¯à¥€",
        "à¤ªà¤¤à¥à¤°", "à¤ªà¤¤à¥à¤°à¤¾à¤šà¤¾à¤°", "à¤¸à¤‚à¤¦à¥‡à¤¶", "à¤¸à¥‚à¤šà¤¨à¤¾", "à¤œà¤¾à¤¨à¤•à¤¾à¤°à¥€",
        "à¤µà¤¿à¤µà¤°à¤£", "à¤µà¤°à¥à¤£à¤¨", "à¤µà¥à¤¯à¤¾à¤–à¥à¤¯à¤¾", "à¤Ÿà¤¿à¤ªà¥à¤ªà¤£à¥€",
    }

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "".join(ch if unicodedata.category(ch) != "Cc" or ch == "\n" else " " for ch in text)
        return text

    def clean_line(self, line: str) -> str:
        line = line.strip()
        line = re.sub(r"^[â€¢\-\*\>\â—¦\d]+[\.\)]?\s*", "", line)
        line = re.sub(r"[\-:.,;]+$", "", line).strip()
        line = " ".join(line.split())
        return line

    def segment_blocks(self, lines, label_map, stop_labels=None):
        blocks = defaultdict(list)
        current_key = None
        lookup = {}
        for key, lbls in label_map.items():
            for lbl in lbls:
                lookup[lbl.lower()] = key
                lookup[lbl.lower().rstrip("- ")] = key

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            # BLOCK TERMINATION: if line contains stop words, close current block
            line_lower = line.lower()
            if any(sw in line_lower for sw in self.BLOCK_STOP_WORDS):
                current_key = None
                continue

            key, inline_value = self._detect_label_in_line(line, lookup)
            if key:
                current_key = key
                if inline_value:
                    blocks[current_key].append(inline_value)
            else:
                if current_key:
                    blocks[current_key].append(raw_line)
        return dict(blocks)

    def _detect_label_in_line(self, line, lookup):
        clean = line.strip().lower()
        clean = re.sub(r"^[^a-zA-Z\u0900-\u097F]+", "", clean)
        clean = re.sub(r"[:.,;\-]+$", "", clean).strip()

        if clean in lookup:
            return lookup[clean], None

        candidates = sorted(lookup.keys(), key=len, reverse=True)
        for lbl in candidates:
            pattern = r"^" + re.escape(lbl) + r"(?:\s*[:.,;\-]\s*|\s+|$)"
            m = re.match(pattern, clean)
            if m:
                remainder = line[m.end():].strip()
                remainder = re.sub(r"^[:.,;\-]+\s*", "", remainder)
                return lookup[lbl], remainder if remainder else None

        # Fallback: concatenated label (no space between label and value)
        for lbl in candidates:
            if clean.startswith(lbl):
                tail = clean[len(lbl):]
                if tail and re.match(r"[a-zA-Z\u0900-\u097F0-9]", tail):
                    remainder = line[len(lbl):].strip()
                    remainder = re.sub(r"^[:.,;\-]+\s*", "", remainder)
                    return lookup[lbl], remainder if remainder else None

        return None, None

    def find_context_window(self, text, keywords, window_chars=300):
        windows = []
        text_lower = text.lower()
        for kw in keywords:
            kw_lower = kw.lower()
            start = 0
            while True:
                idx = text_lower.find(kw_lower, start)
                if idx == -1:
                    break
                w_start = max(0, idx - window_chars)
                w_end = min(len(text), idx + len(kw) + window_chars)
                windows.append((w_start, w_end, text[w_start:w_end]))
                start = idx + len(kw)
        windows.sort()
        merged = []
        for w in windows:
            if merged and w[0] < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], w[1]),
                              text[merged[-1][0]:max(merged[-1][1], w[1])])
            else:
                merged.append(w)
        return merged

    def extract_near_keyword(self, text, keywords, pattern, max_matches=3):
        results = []
        windows = self.find_context_window(text, keywords, window_chars=250)
        for _, _, window in windows:
            matches = re.findall(pattern, window, re.IGNORECASE)
            for m in matches:
                val = m[0] if isinstance(m, tuple) else m
                results.append((val.strip(), window))
                if len(results) >= max_matches:
                    return results
        return results

    def is_valid_name(self, text):
        if len(text) < self.MIN_NAME_LENGTH:
            return False
        if len(text) > self.MAX_NAME_LENGTH:
            return False
        words = text.split()
        if len(words) > self.MAX_NAME_WORDS:
            return False

        lower = text.lower()
        word_set = set(lower.split())
        if word_set & self.GARBAGE_WORDS:
            return False

        hindi_legal_count = sum(1 for w in word_set if w in self.GARBAGE_WORDS)
        if hindi_legal_count >= 2:
            return False

        if re.match(r"^[\d\s\-.,]+$", text):
            return False
        if re.match(r"^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}$", text):
            return False
        if re.match(r"^(rs\.?|â‚¹)?\s*\d", text, re.IGNORECASE):
            return False
        if not re.search(r"[a-zA-Z\u0900-\u097F]", text):
            return False

        allowed = re.sub(r"[a-zA-Z\s\.\-\u0900-\u097F]", "", text)
        if len(allowed) > 2:
            return False

        if len(words) == 1:
            if re.search(r"[\u0900-\u097F]", text):
                return True
            if len(words[0]) < 4:
                return False

        return True

    def is_valid_amount(self, text):
        if not text or len(text) < 2:
            return False
        digits = re.sub(r"[^0-9]", "", text)
        if len(digits) < 3:
            return False
        if len(digits) > 12:  # Reject absurdly large numbers
            return False
        return True

    def is_valid_date(self, text):
        patterns = [
            r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}",
            r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}",
            r"\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}",
        ]
        return any(re.match(p, text.strip()) for p in patterns)

    def clean_value(self, value):
        if isinstance(value, tuple):
            value = value[0] if value else ""
        if not value:
            return ""
        value = str(value).strip()
        value = re.sub(r"[;:,]$", "", value)
        return value.strip()

    def normalize_for_dedup(self, name):
        n = name.lower().strip()
        n = " ".join(n.split())
        n = re.sub(r"^(mr\.?|mrs\.?|ms\.?|shri|smt\.?|shrimati|kumari|miss)\s+", "", n)
        return n

    def score_confidence(self, found_fields, total_fields, has_critical=False,
                         text_length=0, corruption_ratio=0.0):
        if total_fields == 0:
            return 0.0
        base = (found_fields / total_fields) * 70
        if has_critical:
            base += 15
        if text_length > 200:
            base += 5
        if text_length < 80:
            base -= 10
        if corruption_ratio > 0.05:
            base -= 10
        return max(0.0, min(100.0, base))

    def _corruption_ratio(self, text):
        non_print = sum(1 for c in text if unicodedata.category(c) == "Cc" and c != "\n")
        return non_print / len(text) if text else 0.0


# ==============================================================================
# DOCUMENT EXTRACTOR
# ==============================================================================

class DocumentExtractor(BaseExtractor):
    NAME = "document"

    LABEL_MAP = {
        "deed_number": ["deed no", "deed number", "deedno", "document no", "document number"],
        "token_number": ["token no", "token number", "tokenno"],
        "registration_number": ["registration number", "reg no", "reg. no", "registration no"],
        "serial_number": ["serial number", "serial no", "s.no", "s. no", "sl no"],
        "registration_date": ["registration date", "date of registration", "à¤¦à¤¿à¤¨à¤¾à¤‚à¤•", "à¤¤à¤¾à¤°à¥€à¤–"],
        "registration_office": ["registration office", "office of the sub registrar", "sub registrar",
                                "registrar office"],
        "book_number": ["book no", "book number"],
        "volume_number": ["volume no", "volume number", "vol no", "vol. no"],
        "page_number": ["page no", "page number"],
        "document_type": ["document type", "type of deed", "deed type", "nature of document"],
    }

    PATTERNS = {
        "deed_number": [
            r"(?:Deed|Document)\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
        ],
        "token_number": [
            r"Token\s*(?:No|Number)\.?\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
        ],
        "registration_number": [
            r"Registration\s*(?:No|Number)\.?\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
            r"Reg\.?\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
        ],
        "serial_number": [
            r"Serial\s*(?:No|Number)\.?\s*[:\-]?\s*([0-9]+)",
            r"S\.No\.?\s*[:\-]?\s*([0-9]+)",
        ],
        "registration_date": [
            r"(?:Registration\s*Date|Date\s*of\s*Registration)[:\-\s]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
            r"à¤¦à¤¿à¤¨à¤¾à¤‚à¤•[:\-\s]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
            r"Date[:\-\s]*([0-9]{2}/[0-9]{2}/[0-9]{4})",
        ],
        "registration_office": [
            r"(?:Registration\s*Office|Office\s*of\s*the\s*Sub\s*Registrar)[:\-\s,]*([A-Za-z\s]{2,40}?)(?:\n|Sub\s*Registrar|District|$)",
        ],
        "book_number": [
            r"Book\s*(?:No|Number)\.?\s*[:\-]?\s*([A-Za-z0-9]+)",
        ],
        "volume_number": [
            r"Volume\s*(?:No|Number)\.?\s*[:\-]?\s*([A-Za-z0-9\-]+)",
        ],
        "page_number": [
            r"Page\s*(?:No|Number)\.?\s*[:\-]?\s*([0-9\-\s]+)",
        ],
        "document_type": [
            r"(?:Document\s*Type|Type\s*of\s*Deed|Deed\s*Type)[:\-\s]*([A-Za-z\s]{3,30}?)(?:\n|$)",
            r"(?:Sale\s*Deed|Gift\s*Deed|Mortgage\s*Deed|Lease\s*Deed|Will|Power\s*of\s*Attorney)",
        ],
    }

    def extract(self, text: str) -> Dict[str, Any]:
        text = self.normalize_text(text)
        lines = text.split("\n")
        result = {}

        blocks = self.segment_blocks(lines, self.LABEL_MAP)
        for field, block_lines in blocks.items():
            val = self._extract_from_block(block_lines, field)
            if val:
                result[field] = ExtractedField(value=val, confidence=85.0, source=self.NAME,
                                                method="block", raw_matches=[val])

        for field, patterns in self.PATTERNS.items():
            if field in result:
                continue
            keywords = self.LABEL_MAP.get(field, [field.replace("_", " ")])
            for pattern in patterns:
                matches = self.extract_near_keyword(text, keywords, pattern, max_matches=2)
                if matches:
                    best = self._pick_best_match(matches, field)
                    if best:
                        result[field] = ExtractedField(value=best, confidence=70.0,
                                                        source=self.NAME, method="regex_context",
                                                        raw_matches=[m[0] for m in matches])
                        break

        if "document_type" not in result:
            dtype = self._infer_document_type(text)
            if dtype:
                result["document_type"] = ExtractedField(value=dtype, confidence=60.0,
                                                          source=self.NAME, method="inference")

        total = len(self.LABEL_MAP)
        found = len(result)
        has_critical = bool(result.get("deed_number") or result.get("registration_number"))
        conf = self.score_confidence(found, total, has_critical, len(text), self._corruption_ratio(text))

        return {
            "fields": {k: v.to_dict() for k, v in result.items()},
            "_confidence": round(conf, 1),
            "_found": found,
            "_total": total,
        }

    def _extract_from_block(self, lines, field):
        """Return the shortest valid candidate (avoids capturing paragraphs)."""
        candidates = []
        for raw in lines:
            cleaned = self.clean_line(raw)
            if not cleaned:
                continue
            if field == "registration_date" and not self.is_valid_date(cleaned):
                m = re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", cleaned)
                if m:
                    candidates.append(m.group(0))
                continue
            if field in ["deed_number", "token_number", "registration_number", "serial_number"]:
                if re.search(r"[A-Za-z0-9]", cleaned) and len(cleaned) < 30:
                    candidates.append(cleaned)
                    continue
            if field in ["registration_office"]:
                if 3 < len(cleaned) < 40 and not cleaned.isdigit():
                    candidates.append(cleaned)
                    continue
            if field == "document_type":
                if 3 < len(cleaned) < 30:
                    candidates.append(cleaned)
                    continue
            if 1 < len(cleaned) < 50:
                candidates.append(cleaned)
        
        # Return shortest valid candidate â€” the actual value is usually shortest
        if candidates:
            return min(candidates, key=len)
        return None

    def _pick_best_match(self, matches, field):
        for val, _ctx in matches:
            val = self.clean_value(val)
            if field == "registration_date" and not self.is_valid_date(val):
                continue
            if field in ["deed_number", "token_number", "registration_number", "serial_number"]:
                if re.search(r"[A-Za-z0-9]", val) and len(val) < 30:
                    return val
            if field == "registration_office":
                if 3 < len(val) < 40 and not val.isdigit():
                    return val
            if len(val) > 1 and len(val) < 50:
                return val
        return None

    def _infer_document_type(self, text):
        text_lower = text.lower()
        type_keywords = {
            "Sale Deed": ["sale deed", "à¤¬à¤¿à¤•à¥à¤°à¥€ à¤ªà¤¤à¥à¤°", "à¤¬à¤¿à¤•à¥à¤°à¥€ à¤¦à¤¸à¥à¤¤à¤¾à¤µà¥‡à¤œà¤¼", "à¤µà¤¿à¤•à¥à¤°à¤¯ à¤µà¤¿à¤²à¥‡à¤–"],
            "Gift Deed": ["gift deed", "à¤­à¥‡à¤‚à¤Ÿ à¤ªà¤¤à¥à¤°", "à¤¹à¤¿à¤¬à¤¾"],
            "Mortgage Deed": ["mortgage deed", "à¤¬à¤‚à¤§à¤• à¤ªà¤¤à¥à¤°", "à¤‡à¤ªà¥‹à¤¥à¤¾"],
            "Lease Deed": ["lease deed", "à¤ªà¤Ÿà¥à¤Ÿà¤¾ à¤ªà¤¤à¥à¤°", "à¤ªà¤Ÿà¥à¤Ÿà¤¾"],
            "Will": ["will", "à¤µà¤¸à¥€à¤¯à¤¤"],
            "Power of Attorney": ["power of attorney", "à¤®à¥à¤–à¥à¤¤à¤¾à¤°à¤¨à¤¾à¤®à¤¾"],
            "Release Deed": ["release deed", "à¤°à¤¿à¤¹à¤¾à¤ˆ à¤ªà¤¤à¥à¤°"],
            "Exchange Deed": ["exchange deed", "à¤µà¤¿à¤¨à¤¿à¤®à¤¯ à¤ªà¤¤à¥à¤°"],
        }
        for dtype, kws in type_keywords.items():
            for kw in kws:
                if kw in text_lower:
                    return dtype
        return None


# ==============================================================================
# FINANCIAL EXTRACTOR
# ==============================================================================

class FinancialExtractor(BaseExtractor):
    NAME = "financial"

    LABEL_MAP = {
        "sale_consideration": ["sale consideration", "consideration amount", "total consideration",
                              "sold for", "purchased for", "price", "amount", "value"],
        "market_value": ["market value", "guidance value", "circle rate", "ready reckoner"],
        "registration_fee": ["registration fee", "reg fee", "registration charges"],
        "stamp_duty": ["stamp duty", "stamp charges"],
        "other_fee": ["other fee", "miscellaneous fee", "additional fee", "processing fee"],
    }

    PATTERNS = {
        "sale_consideration": [
            r"(?:sale\s+consideration|consideration\s+amount|total\s+consideration)[\s:\-]*(?:Rs\.?|INR|â‚¹|à¤°à¥\.?|à¤°à¥à¤ªà¤¯à¥‡)?\s*([0-9,\.]+)",
            r"(?:sold|bought|purchased)\s+(?:for|at)\s+(?:a\s+)?(?:sum\s+of\s+)?(?:Rs\.?|INR|â‚¹|à¤°à¥\.?|à¤°à¥à¤ªà¤¯à¥‡)?\s*([0-9,\.]+)",
            r"(?:price|amount|value)\s+(?:of|is)\s+(?:Rs\.?|INR|â‚¹|à¤°à¥\.?|à¤°à¥à¤ªà¤¯à¥‡)?\s*([0-9,\.]+)",
            r"(?:Rs\.?|INR|â‚¹|à¤°à¥\.?|à¤°à¥à¤ªà¤¯à¥‡)\s*([0-9,\.]+)\s*(?:only)?\s*(?:as|being)?\s*(?:the\s+)?(?:sale\s+)?(?:consideration|price|amount)",
        ],
        "market_value": [
            r"(?:market\s+value|guidance\s+value|circle\s+rate)[\s:\-]*(?:Rs\.?|INR|â‚¹|à¤°à¥\.?|à¤°à¥à¤ªà¤¯à¥‡)?\s*([0-9,\.]+)",
        ],
        "registration_fee": [
            r"(?:registration\s+fee|reg\.\s*fee)[\s:\-]*(?:Rs\.?|INR|â‚¹|à¤°à¥\.?|à¤°à¥à¤ªà¤¯à¥‡)?\s*([0-9,\.]+)",
        ],
        "stamp_duty": [
            r"(?:stamp\s+duty)[\s:\-]*(?:Rs\.?|INR|â‚¹|à¤°à¥\.?|à¤°à¥à¤ªà¤¯à¥‡)?\s*([0-9,\.]+)",
        ],
        "other_fee": [
            r"(?:other\s+fee|miscellaneous\s+fee|additional\s+fee)[\s:\-]*(?:Rs\.?|INR|â‚¹|à¤°à¥\.?|à¤°à¥à¤ªà¤¯à¥‡)?\s*([0-9,\.]+)",
        ],
    }

    def extract(self, text: str) -> Dict[str, Any]:
        text = self.normalize_text(text)
        lines = text.split("\n")
        result = {}

        blocks = self.segment_blocks(lines, self.LABEL_MAP)
        for field, block_lines in blocks.items():
            val = self._extract_amount_from_block(block_lines)
            if val:
                result[field] = ExtractedField(value=val, confidence=85.0, source=self.NAME,
                                                method="block", raw_matches=[val])

        for field, patterns in self.PATTERNS.items():
            if field in result:
                continue
            keywords = self.LABEL_MAP.get(field, [field.replace("_", " ")])
            for pattern in patterns:
                matches = self.extract_near_keyword(text, keywords, pattern, max_matches=3)
                if matches:
                    best = self._pick_best_amount(matches)
                    if best:
                        result[field] = ExtractedField(value=best, confidence=70.0,
                                                        source=self.NAME, method="regex_context",
                                                        raw_matches=[m[0] for m in matches])
                        break

        if "sale_consideration" not in result:
            candidates = self._scan_global_amounts(text)
            if candidates:
                # Pick the largest amount that looks like a sale consideration
                reasonable = [c for c in candidates if 1000 <= c["normalized"] <= 999999999]
                if reasonable:
                    best = max(reasonable, key=lambda x: x["normalized"])
                    result["sale_consideration"] = ExtractedField(value=best["raw"], confidence=55.0,
                                                                   source=self.NAME, method="global_scan",
                                                                   raw_matches=[best["raw"]])

        total = len(self.LABEL_MAP)
        found = len(result)
        has_critical = bool(result.get("sale_consideration"))
        conf = self.score_confidence(found, total, has_critical, len(text), self._corruption_ratio(text))

        return {
            "fields": {k: v.to_dict() for k, v in result.items()},
            "_confidence": round(conf, 1),
            "_found": found,
            "_total": total,
        }

    def _extract_amount_from_block(self, lines):
        for raw in lines:
            cleaned = self.clean_line(raw)
            if not cleaned:
                continue
            m = re.search(r"(?:Rs\.?|INR|â‚¹|à¤°à¥\.?|à¤°à¥à¤ªà¤¯à¥‡|à¤°à¥à¤ªà¤¯à¤¾)?\s*([0-9,\.]+)", cleaned, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if self.is_valid_amount(val):
                    return val
            if re.match(r"^[0-9,\.]+$", cleaned):
                if self.is_valid_amount(cleaned):
                    return cleaned
        return None

    def _pick_best_amount(self, matches):
        best = None
        best_num = 0.0
        for val, _ctx in matches:
            val = self.clean_value(val)
            if not self.is_valid_amount(val):
                continue
            num_str = val.replace(",", "").replace(" ", "")
            try:
                num = float(num_str)
                if num > best_num:
                    best_num = num
                    best = val
            except ValueError:
                if best is None:
                    best = val
        return best

    def _scan_global_amounts(self, text):
        pattern = r"(?:Rs\.?|INR|â‚¹|à¤°à¥\.?|à¤°à¥à¤ªà¤¯à¥‡|à¤°à¥à¤ªà¤¯à¤¾)?\s*([0-9,\.]+)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        candidates = []
        for m in matches:
            clean = self.clean_value(m)
            if not self.is_valid_amount(clean):
                continue
            num_str = clean.replace(",", "").replace(" ", "")
            try:
                num = float(num_str)
                candidates.append({"raw": clean, "normalized": num})
            except ValueError:
                pass
        seen = set()
        unique = []
        for c in sorted(candidates, key=lambda x: x["normalized"], reverse=True):
            if c["raw"] not in seen:
                seen.add(c["raw"])
                unique.append(c)
        return unique


# ==============================================================================
# PROPERTY EXTRACTOR
# ==============================================================================

class PropertyExtractor(BaseExtractor):
    NAME = "property"

    # Hindi labels removed from block map to avoid body-text false positives.
    LABEL_MAP = {
        "district": ["district"],
        "village": ["village", "mouza"],
        "tehsil": ["tehsil", "taluka"],
        "survey_number": ["survey no", "survey number"],
        "plot_number": ["plot no", "plot number"],
        "khasra_number": ["khasra no", "khasra number"],
        "khata_number": ["khata no", "khata number", "khatian no"],
        "area": ["area", "land area", "extent"],
    }

    PATTERNS = {
        "district": [
            r"District[\s:\-]*([A-Za-z\s]{2,30}?)(?:\n|Village|Tehsil|Taluka|Sub[-\s]?District|$)",
            r"à¤œà¤¿à¤²à¤¾[\s:\-]*([\u0900-\u097F\s]{2,30}?)(?:\n|à¤—à¤¾à¤à¤µ|à¤—à¤¾à¤‚à¤µ|à¤¤à¤¹à¤¸à¥€à¤²|à¤¤à¤¾à¤²à¥à¤•à¤¾|$)",
        ],
        "village": [
            r"Village[\s:\-]*([A-Za-z\s]{2,30}?)(?:\n|District|Tehsil|Taluka|Survey|Plot|Khasra|Khata|$)",
            r"(?:à¤—à¤¾à¤à¤µ|à¤—à¤¾à¤‚à¤µ)[\s:\-]*([\u0900-\u097F\s]{2,30}?)(?:\n|à¤œà¤¿à¤²à¤¾|à¤¤à¤¹à¤¸à¥€à¤²|à¤¤à¤¾à¤²à¥à¤•à¤¾|à¤¸à¤°à¥à¤µà¥‡|à¤ªà¥à¤²à¥‰à¤Ÿ|à¤–à¤¸à¤°à¤¾|à¤–à¤¾à¤¤à¤¾|$)",
            r"Mouza[\s:\-]*([A-Za-z\s]{2,30}?)(?:\n|District|Tehsil|J\.L\.|$)",
        ],
        "tehsil": [
            r"(?:Tehsil|Taluka)[\s:\-]*([A-Za-z\s]{2,30}?)(?:\n|District|Village|Sub[-\s]?District|$)",
            r"(?:à¤¤à¤¹à¤¸à¥€à¤²|à¤¤à¤¾à¤²à¥à¤•à¤¾)[\s:\-]*([\u0900-\u097F\s]{2,30}?)(?:\n|à¤œà¤¿à¤²à¤¾|à¤—à¤¾à¤à¤µ|à¤—à¤¾à¤‚à¤µ|$)",
        ],
        "survey_number": [
            r"(?:Survey\s*No|Survey\s*Number)[\s:\-]*([0-9A-Z\-/]+)",
            r"(?:à¤¸à¤°à¥à¤µà¥‡\s*à¤¸à¤‚à¤–à¥à¤¯à¤¾|à¤¸à¤°à¥à¤µà¥‡\s*à¤¨à¤‚)[\s:\-]*([0-9A-Z\-/]+)",
        ],
        "plot_number": [
            r"(?:Plot\s*No|Plot\s*Number)[\s:\-]*([0-9A-Z\-/]+)",
            r"(?:à¤ªà¥à¤²à¥‰à¤Ÿ\s*à¤¨à¤‚|à¤ªà¥à¤²à¥‰à¤Ÿ\s*à¤¸à¤‚à¤–à¥à¤¯à¤¾)[\s:\-]*([0-9A-Z\-/]+)",
        ],
        "khasra_number": [
            r"(?:Khasra\s*No|Khasra\s*Number)[\s:\-]*([0-9A-Z\-/]+)",
            r"(?:à¤–à¤¸à¤°à¤¾\s*à¤¨à¤‚|à¤–à¤¸à¤°à¤¾\s*à¤¸à¤‚à¤–à¥à¤¯à¤¾)[\s:\-]*([0-9A-Z\-/]+)",
        ],
        "khata_number": [
            r"(?:Khata\s*No|Khata\s*Number|Khatian\s*No)[\s:\-]*([0-9A-Z\-/]+)",
            r"(?:à¤–à¤¾à¤¤à¤¾\s*à¤¨à¤‚|à¤–à¤¾à¤¤à¤¾\s*à¤¸à¤‚à¤–à¥à¤¯à¤¾)[\s:\-]*([0-9A-Z\-/]+)",
        ],
        "area": [
            r"(?:Area|Land\s*Area)[\s:\-]*([0-9\.,]+\s*(?:sq\.?\s*ft|sq\.?\s*yd|acre|bigha|kattha|decimal|hectare|ha|ftÂ²|mÂ²))",
            r"(?:à¤•à¥à¤·à¥‡à¤¤à¥à¤°à¤«à¤²|à¤•à¥à¤·à¥‡à¤¤à¥à¤°|à¤°à¤•à¤¬à¤¾)[\s:\-]*([0-9\.,]+\s*(?:à¤µà¤°à¥à¤—|à¤µà¤°à¥à¤—à¤«à¥à¤Ÿ|à¤µà¤°à¥à¤—à¤®à¥€à¤Ÿà¤°|à¤à¤•à¤¡à¤¼|à¤¬à¥€à¤˜à¤¾|à¤•à¤Ÿà¥à¤ à¤¾|à¤¡à¥‡à¤¸à¥€à¤®à¤²|à¤¹à¥‡à¤•à¥à¤Ÿà¥‡à¤¯à¤°))",
        ],
    }

    def extract(self, text: str) -> Dict[str, Any]:
        text = self.normalize_text(text)
        lines = text.split("\n")
        result = {}

        blocks = self.segment_blocks(lines, self.LABEL_MAP)
        for field, block_lines in blocks.items():
            val = self._extract_from_block(block_lines, field)
            if val:
                result[field] = ExtractedField(value=val, confidence=85.0, source=self.NAME,
                                                method="block", raw_matches=[val])

        for field, patterns in self.PATTERNS.items():
            if field in result:
                continue
            keywords = self.LABEL_MAP.get(field, [field.replace("_", " ")])
            for pattern in patterns:
                matches = self.extract_near_keyword(text, keywords, pattern, max_matches=2)
                if matches:
                    best = self._pick_best_match(matches, field)
                    if best:
                        result[field] = ExtractedField(value=best, confidence=70.0,
                                                        source=self.NAME, method="regex_context",
                                                        raw_matches=[m[0] for m in matches])
                        break

        total = len(self.LABEL_MAP)
        found = len(result)
        has_critical = bool(result.get("district") or result.get("village"))
        conf = self.score_confidence(found, total, has_critical, len(text), self._corruption_ratio(text))

        return {
            "fields": {k: v.to_dict() for k, v in result.items()},
            "_confidence": round(conf, 1),
            "_found": found,
            "_total": total,
        }

    def _extract_from_block(self, lines, field):
        for raw in lines:
            cleaned = self.clean_line(raw)
            if not cleaned:
                continue
            if field in ["survey_number", "plot_number", "khasra_number", "khata_number"]:
                if re.search(r"[0-9A-Z]", cleaned) and len(cleaned) < 25:
                    return cleaned
            if field == "area":
                if re.search(r"[0-9]", cleaned) and len(cleaned) < 30:
                    return cleaned
            if field in ["district", "village", "tehsil"]:
                if 2 < len(cleaned) < 35 and not cleaned.isdigit():
                    return cleaned
            if 1 < len(cleaned) < 40:
                return cleaned
        return None

    def _pick_best_match(self, matches, field):
        for val, _ctx in matches:
            val = self.clean_value(val)
            if field in ["survey_number", "plot_number", "khasra_number", "khata_number"]:
                if re.search(r"[0-9A-Z]", val) and len(val) < 25:
                    return val
            if field == "area":
                if re.search(r"[0-9]", val) and len(val) < 30:
                    return val
            if field in ["district", "village", "tehsil"]:
                if 2 < len(val) < 35 and not val.isdigit():
                    return val
            if len(val) > 1 and len(val) < 40:
                return val
        return None


# ==============================================================================
# PARTY EXTRACTOR
# ==============================================================================

class PartyExtractor(BaseExtractor):
    NAME = "party"

    OUTPUT_KEYS = [
        "executed_by", "presented_by", "vendor", "seller", "buyer",
        "purchaser", "vendee", "transferor", "transferee", "executant",
        "presenter", "identifier", "witness", "claimant"
    ]

    # Each label appears in EXACTLY ONE key. Removed Hindi body words.
    LABEL_MAP = {
        "executed_by": [
            "executed by", "executedby", "executed-by",
        ],
        "presented_by": [
            "presented by", "presentedby", "presented-by",
        ],
        "vendor": [
            "vendor", "vendor-", "vendor / seller", "vendor/seller",
        ],
        "seller": [
            "seller", "seller-",
        ],
        "buyer": [
            "buyer", "buyer-", "in favour of", "in favor of",
            "in-favour-of", "in-favor-of",
        ],
        "purchaser": [
            "purchaser", "purchaser-",
        ],
        "vendee": [
            "vendee", "vendee-",
        ],
        "transferor": [
            "transferor", "transferor-",
        ],
        "transferee": [
            "transferee", "transferee-",
        ],
        "executant": [
            "executant", "executant-",
        ],
        "presenter": [
            "presenter", "presenter-",
        ],
        "identifier": [
            "identifier", "identifier-",
        ],
        "witness": [
            "witness", "witness-", "witnesses", "witnesses-",
        ],
        "claimant": [
            "claimant", "claimant-",
        ],
    }

    # Safe Hindi equivalents (header-only, not body text)
    HINDI_LABEL_MAP = {
        "presented_by": ["à¤ªà¥à¤°à¤¸à¥à¤¤à¥à¤¤à¤•à¤°à¥à¤¤à¤¾", "à¤ªà¥à¤°à¤¸à¥à¤¤à¥à¤¤à¤•à¤°à¥à¤¤à¤¾ à¤¦à¥à¤µà¤¾à¤°à¤¾"],
        "witness": ["à¤—à¤µà¤¾à¤¹", "à¤—à¤µà¤¾à¤¹à¥‹à¤‚"],
        "buyer": ["à¤–à¤°à¥€à¤¦à¤¾à¤°"],
    }

    def extract(self, text: str) -> Dict[str, Any]:
        text = self.normalize_text(text)
        lines = text.split("\n")

        merged_labels = dict(self.LABEL_MAP)
        for key, labels in self.HINDI_LABEL_MAP.items():
            if key not in merged_labels:
                merged_labels[key] = []
            merged_labels[key].extend(labels)

        blocks = self.segment_blocks(lines, merged_labels)

        result = {key: [] for key in self.OUTPUT_KEYS}
        seen_names = defaultdict(set)

        for key, block_lines in blocks.items():
            names = self._extract_names_from_block(block_lines)
            for name in names:
                norm = self.normalize_for_dedup(name)
                if norm not in seen_names[key]:
                    seen_names[key].add(norm)
                    result[key].append(name)

        has_seller = bool(result.get("seller") or result.get("vendor") or result.get("executed_by"))
        has_buyer = bool(result.get("buyer") or result.get("purchaser") or result.get("vendee"))
        total_names = sum(len(v) for v in result.values())
        num_blocks = len(blocks)

        score = 0
        if num_blocks > 0:
            score += 40 + min(num_blocks * 8, 40)
        if total_names > 0:
            score += min(total_names * 4, 20)
        else:
            score -= 25
        if has_seller and has_buyer:
            score += 10
        if len(text.strip()) < 80:
            score -= 10
        if self._corruption_ratio(text) > 0.05:
            score -= 10

        return {
            "fields": result,
            "_confidence": max(0, min(100, score)),
            "_found": num_blocks,
            "_total": len(self.OUTPUT_KEYS),
        }

    def _extract_names_from_block(self, block_lines):
        names = []
        seen = set()
        for raw in block_lines:
            cleaned = self.clean_line(raw)
            if not cleaned:
                continue
            candidates = self._split_name_line(cleaned)
            for cand in candidates:
                cand = cand.strip()
                if not cand:
                    continue
                if self.is_valid_name(cand):
                    norm = self.normalize_for_dedup(cand)
                    if norm not in seen:
                        seen.add(norm)
                        names.append(cand)
        return names

    def _split_name_line(self, text):
        if "," not in text and " and " not in text and " & " not in text and " à¤”à¤° " not in text:
            return [text]
        parts = re.split(r"\s*(?:,|\s+and\s+|\s+&\s+|\s+à¤”à¤°\s+)\s*", text)
        if any(len(p.strip()) < self.MIN_NAME_LENGTH for p in parts):
            return [text]
        return [p.strip() for p in parts if p.strip()]


# ==============================================================================
# EXTRACTION ENGINE (Orchestrator)
# ==============================================================================

class ExtractionEngine:
    def __init__(self):
        self.document_extractor = DocumentExtractor()
        self.financial_extractor = FinancialExtractor()
        self.property_extractor = PropertyExtractor()
        self.party_extractor = PartyExtractor()

    def run(self, ocr_text: str) -> Dict[str, Any]:
        if not ocr_text or not isinstance(ocr_text, str):
            return self._empty_result()

        doc_result = self.document_extractor.extract(ocr_text)
        fin_result = self.financial_extractor.extract(ocr_text)
        prop_result = self.property_extractor.extract(ocr_text)
        party_result = self.party_extractor.extract(ocr_text)

        result = {
            "document_details": {
                **{k: v for k, v in doc_result.get("fields", {}).items()},
                "_confidence": doc_result.get("_confidence", 0.0),
                "_found": doc_result.get("_found", 0),
                "_total": doc_result.get("_total", 0),
            },
            "financial": {
                **{k: v for k, v in fin_result.get("fields", {}).items()},
                "_confidence": fin_result.get("_confidence", 0.0),
                "_found": fin_result.get("_found", 0),
                "_total": fin_result.get("_total", 0),
            },
            "property": {
                **{k: v for k, v in prop_result.get("fields", {}).items()},
                "_confidence": prop_result.get("_confidence", 0.0),
                "_found": prop_result.get("_found", 0),
                "_total": prop_result.get("_total", 0),
            },
            "parties": {
                **{k: v for k, v in party_result.get("fields", {}).items()},
                "_confidence": party_result.get("_confidence", 0.0),
                "_found": party_result.get("_found", 0),
                "_total": party_result.get("_total", 0),
            },
        }

        section_confs = [
            result["document_details"]["_confidence"],
            result["financial"]["_confidence"],
            result["property"]["_confidence"],
            result["parties"]["_confidence"],
        ]
        weights = [0.20, 0.20, 0.20, 0.40]
        overall = sum(c * w for c, w in zip(section_confs, weights))
        result["overall_confidence"] = round(overall, 1)

        result["metadata"] = {
            "engine_version": "2.1.1",
            "extractors_used": ["document", "financial", "property", "party"],
            "text_length": len(ocr_text),
            "lines_processed": len(ocr_text.split("\n")),
        }

        return result

    def run_json(self, ocr_text: str) -> str:
        return json.dumps(self.run(ocr_text), ensure_ascii=False, indent=2)

    def _empty_result(self):
        return {
            "document_details": {"_confidence": 0.0, "_found": 0, "_total": 0},
            "financial": {"_confidence": 0.0, "_found": 0, "_total": 0},
            "property": {"_confidence": 0.0, "_found": 0, "_total": 0},
            "parties": {"_confidence": 0.0, "_found": 0, "_total": 0},
            "overall_confidence": 0.0,
            "metadata": {},
        }
