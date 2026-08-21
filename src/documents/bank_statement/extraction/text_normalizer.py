"""
Bank Statement Text Normalizer.

Phase 2 - Document Intelligence / Extraction.

Responsibilities:
- accept text produced by native PDF extraction or OCR
- normalize Unicode and whitespace safely
- preserve meaningful line boundaries
- remove common extraction noise
- produce a consistent text representation for downstream parsers
- remain completely bank-independent

Important:
Normalization must be conservative.

Transaction parsers may depend on:
- line boundaries
- dates
- amounts
- reference numbers
- debit/credit indicators

Therefore this module must NOT aggressively rewrite content.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NormalizedTextResult:
    text: str
    line_count: int
    char_count: int
    original_char_count: int
    removed_empty_lines: int

    def to_dict(self) -> dict:
        return asdict(self)


class TextNormalizer:
    """
    Normalize extracted bank-statement text into a stable,
    bank-independent representation.
    """

    # Unicode characters commonly produced by PDF/OCR engines.
    SPACE_TRANSLATION = {
        "\u00a0": " ",   # non-breaking space
        "\u2000": " ",
        "\u2001": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u205f": " ",
        "\u3000": " ",
    }

    DASH_TRANSLATION = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }

    QUOTE_TRANSLATION = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def normalize(
        self,
        text: str,
    ) -> NormalizedTextResult:

        if text is None:
            raise ValueError(
                "Text cannot be None."
            )

        if not isinstance(text, str):
            raise TypeError(
                "Text must be a string."
            )

        original_char_count = len(text)

        if not text:
            return NormalizedTextResult(
                text="",
                line_count=0,
                char_count=0,
                original_char_count=0,
                removed_empty_lines=0,
            )

        text = self._normalize_unicode(
            text
        )

        text = self._normalize_line_endings(
            text
        )

        text = self._remove_control_characters(
            text
        )

        text = self._normalize_special_characters(
            text
        )

        normalized_lines: list[str] = []

        empty_line_run = 0
        removed_empty_lines = 0

        for raw_line in text.split("\n"):

            line = self._normalize_line(
                raw_line
            )

            if not line:

                empty_line_run += 1

                # Keep at most one blank line so that page/
                # section separation is not completely lost.
                if empty_line_run <= 1:
                    normalized_lines.append("")
                else:
                    removed_empty_lines += 1

                continue

            empty_line_run = 0

            normalized_lines.append(
                line
            )

        normalized_text = "\n".join(
            normalized_lines
        ).strip()

        line_count = sum(
            1
            for line in normalized_text.splitlines()
            if line.strip()
        )

        return NormalizedTextResult(
            text=normalized_text,
            line_count=line_count,
            char_count=len(normalized_text),
            original_char_count=original_char_count,
            removed_empty_lines=removed_empty_lines,
        )

    # --------------------------------------------------------
    # Unicode normalization
    # --------------------------------------------------------

    @staticmethod
    def _normalize_unicode(
        text: str,
    ) -> str:
        """
        Normalize equivalent Unicode forms while preserving
        readable text.
        """

        return unicodedata.normalize(
            "NFKC",
            text,
        )

    # --------------------------------------------------------
    # Line endings
    # --------------------------------------------------------

    @staticmethod
    def _normalize_line_endings(
        text: str,
    ) -> str:

        return (
            text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

    # --------------------------------------------------------
    # Control characters
    # --------------------------------------------------------

    @staticmethod
    def _remove_control_characters(
        text: str,
    ) -> str:
        """
        Remove non-printable control characters except:
        newline and tab.
        """

        cleaned_chars = []

        for char in text:

            if char in {
                "\n",
                "\t",
            }:
                cleaned_chars.append(char)
                continue

            category = unicodedata.category(
                char
            )

            if category.startswith("C"):
                continue

            cleaned_chars.append(char)

        return "".join(
            cleaned_chars
        )

    # --------------------------------------------------------
    # Character normalization
    # --------------------------------------------------------

    def _normalize_special_characters(
        self,
        text: str,
    ) -> str:

        for source, target in (
            self.SPACE_TRANSLATION.items()
        ):
            text = text.replace(
                source,
                target,
            )

        for source, target in (
            self.DASH_TRANSLATION.items()
        ):
            text = text.replace(
                source,
                target,
            )

        for source, target in (
            self.QUOTE_TRANSLATION.items()
        ):
            text = text.replace(
                source,
                target,
            )

        return text

    # --------------------------------------------------------
    # Individual line normalization
    # --------------------------------------------------------

    @staticmethod
    def _normalize_line(
        line: str,
    ) -> str:
        """
        Normalize one line without destroying transaction
        structure.
        """

        if not line:
            return ""

        # Tabs from OCR/PDF extraction are converted to spaces.
        line = line.replace(
            "\t",
            " ",
        )

        # Remove null bytes defensively.
        line = line.replace(
            "\x00",
            "",
        )

        # Collapse repeated horizontal whitespace.
        line = re.sub(
            r"[ ]{2,}",
            " ",
            line,
        )

        return line.strip()


text_normalizer = TextNormalizer()