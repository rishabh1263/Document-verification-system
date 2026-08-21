"""
src/field_extractor.py

Structured field extraction for Indian Driving Licences.

Designed for PaddleOCR output.

Extracts:
- DL number
- Name
- Relative name
- DOB
- Date of issue
- Non-transport validity
- Transport validity
- Blood group
- Multiple classes of vehicle
- Address
- PIN code
"""

import re
from datetime import datetime


# ============================================================
# CONSTANTS
# ============================================================

VALID_BLOOD_GROUPS = {
    "A+",
    "A-",
    "B+",
    "B-",
    "AB+",
    "AB-",
    "O+",
    "O-",
}


VALID_COV_CODES = {
    "LMV",
    "MCWG",
    "MCWOG",
    "HMV",
    "HGMV",
    "MGV",
    "LMV-NT",
    "LMV-TR",
    "TRANS",
    "PSV",
}


DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})\b"
)


PIN_PATTERN = re.compile(
    r"\b([1-9][0-9]{5})\b"
)


RELATIONSHIP_PATTERN = re.compile(
    r"("
    r"S\s*/\s*D\s*/\s*M?\s*W"
    r"|S\s*/\s*D\s*/\s*W"
    r"|S\s*/\s*D\s*/\s*H"
    r"|S\s*/\s*O"
    r"|D\s*/\s*O"
    r"|W\s*/\s*O"
    r"|C\s*/\s*O"
    r"|SON\s+OF"
    r"|DAUGHTER\s+OF"
    r"|WIFE\s+OF"
    r")",
    flags=re.IGNORECASE,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def _text(line):
    return str(
        line.get("text", "")
    ).strip()


def _confidence(line):
    try:
        return float(
            line.get("confidence", 0.0)
        )
    except (TypeError, ValueError):
        return 0.0


def _bounds(line):
    bbox = line.get("bbox")

    if not bbox:
        return None

    try:
        xs = [
            float(point[0])
            for point in bbox
        ]

        ys = [
            float(point[1])
            for point in bbox
        ]

        return (
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )

    except (
        TypeError,
        ValueError,
        IndexError,
    ):
        return None


def _sort_lines(lines):

    def key(line):

        bounds = _bounds(line)

        if bounds is None:
            return 0, 0

        x1, y1, _, _ = bounds

        return y1, x1

    return sorted(
        lines,
        key=key,
    )


def _clean_spaces(text):
    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()


def _normalise_date(value):

    if not value:
        return None

    value = (
        value
        .replace("/", "-")
        .replace(".", "-")
    )

    return value


def _extract_date(text):
    """
    Extract a date from OCR text.

    Handles dates embedded inside other OCR text, for example:

        DOB:03-01-1994BG:
        DOI:11-10-2012
        Valid Till:10-10-2032(NT)
        MCWG04-10-2022
    """

    if not text:
        return None

    text = str(text).strip()

    # --------------------------------------------------------
    # OCR NORMALIZATION
    # --------------------------------------------------------

    normalized = (
        text
        .replace("â€“", "-")
        .replace("â€”", "-")
        .replace("_", "-")
    )

    # --------------------------------------------------------
    # FIND DATE ANYWHERE IN STRING
    # --------------------------------------------------------
    #
    # Do NOT use \b around the date.
    #
    # In:
    #
    #     DOB:03-01-1994BG:
    #
    # there is no word boundary between "4" and "B",
    # because both digits and letters are word characters.
    # --------------------------------------------------------

    match = re.search(
        r"(?<!\d)"
        r"([0-3]?\d)"
        r"[-/.]"
        r"([01]?\d)"
        r"[-/.]"
        r"((?:19|20)\d{2})"
        r"(?!\d)",
        normalized,
    )

    if not match:
        return None

    day = int(match.group(1))
    month = int(match.group(2))
    year = int(match.group(3))

    # --------------------------------------------------------
    # REAL CALENDAR VALIDATION
    # --------------------------------------------------------

    try:
        parsed = datetime(year, month, day)
    except ValueError:
        return None

    # --------------------------------------------------------
    # STANDARD OUTPUT
    # --------------------------------------------------------

    return parsed.strftime("%d-%m-%Y")


def _date_tuple(value):
    """
    Parse DD-MM-YYYY into an integer tuple (year, month, day)
    for reliable comparison.
    """
    if not value:
        return None
    try:
        d, m, y = value.split("-")
        return (int(y), int(m), int(d))
    except ValueError:
        return None


# ============================================================
# DL NUMBER
# ============================================================

def _normalise_dl_number(text):

    if not text:
        return None

    value = str(text).upper()

    value = re.sub(
        r"\bDL\s*(?:NO|NUMBER)?\s*[:.\-]?",
        " ",
        value,
    )

    value = _clean_spaces(
        value
    )

    # Example:
    #
    # MH03 20220045390
    #
    # MH + 03 + 2022 + 0045390

    match = re.search(
        r"\b([A-Z]{2})\s*[- ]?"
        r"(\d{2})\s*[- ]?"
        r"(\d{4})\s*[- ]?"
        r"(\d{7})\b",
        value,
    )

    if match:

        return (
            match.group(1)
            + match.group(2)
            + match.group(3)
            + match.group(4)
        )

    # Already compact.
    match = re.search(
        r"\b([A-Z]{2}\d{13})\b",
        value,
    )

    if match:
        return match.group(1)

    return None


def _find_dl_number(lines):

    candidates = []

    for line in lines:

        candidate = _normalise_dl_number(
            _text(line)
        )

        if candidate:

            candidates.append(
                (
                    candidate,
                    _confidence(line),
                )
            )

    if not candidates:

        return None, []

    candidates.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    best = candidates[0][0]

    others = []

    for candidate, _ in candidates:

        if (
            candidate != best
            and candidate not in others
        ):
            others.append(candidate)

    return best, others


# ============================================================
# DOB
# ============================================================

def _find_dob(lines, max_date=None):
    """
    Extract Date of Birth.

    Tolerates common OCR misreads of the DOB label and searches
    both forward and backward from the label.

    If *max_date* is provided (typically the Date of Issue), any
    candidate date on or after that value is rejected.  This
    prevents COV/validity dates from being mistaken for DOB.
    """

    def _is_plausible(candidate_date):
        if not max_date:
            return True
        c = _date_tuple(candidate_date)
        r = _date_tuple(max_date)
        if c is None or r is None:
            return True
        return c < r

    dob_pattern = re.compile(
        r"\b("
        r"DOB"
        r"|D[O0]B"
        r"|D\.O\.B\.?"
        r"|DATE\s+OF\s+BIRTH"
        r"|BIRTH\s+DATE"
        r"|BORN"
        r")\b",
        flags=re.IGNORECASE,
    )

    # Candidate line itself must not contain these.
    forbidden_pattern = re.compile(
        r"\b("
        r"D[O0][I1]"
        r"|DATE\s+OF\s+ISSUE"
        r"|ISSU"
        r"|VALID"
        r"|VALIDITY"
        r"|COV"
        r"|CLASS\s+OF\s+VEHICLE"
        r"|BG"
        r"|BLOOD"
        r"|PIN"
        r"|ADDRESS"
        r"|ADD\b"
        r"|NAME"
        r"|LMV"
        r"|MCWG"
        r"|MCWOG"
        r"|HMV"
        r"|HGMV"
        r"|MGV"
        r"|LMV-NT"
        r"|LMV-TR"
        r"|TRANS"
        r"|PSV"
        r")\b",
        flags=re.IGNORECASE,
    )

    # Reject candidates that sit on a line with a COV code
    # (or immediately next to one) because those are COV dates.
    cov_pattern = re.compile(
        r"\b("
        r"LMV"
        r"|MCWG"
        r"|MCWOG"
        r"|HMV"
        r"|HGMV"
        r"|MGV"
        r"|LMV-NT"
        r"|LMV-TR"
        r"|TRANS"
        r"|PSV"
        r")\b",
        flags=re.IGNORECASE,
    )

    def _line_has_cov_context(idx):
        """True if this line or an immediate neighbour contains a COV code."""
        for off in (0, -1, 1):
            j = idx + off
            if 0 <= j < len(lines):
                if cov_pattern.search(_text(lines[j])):
                    return True
        return False

    # --------------------------------------------------------
    # 1. SAME LINE
    # --------------------------------------------------------

    for line in lines:

        text = _text(line)

        if not dob_pattern.search(text):
            continue

        date = _extract_date(text)

        if date and _is_plausible(date):
            return date

    # --------------------------------------------------------
    # 2. CONTROLLED NEARBY SEARCH
    # --------------------------------------------------------

    for index, line in enumerate(lines):

        text = _text(line)

        if not dob_pattern.search(text):
            continue

        for offset in (-2, -1, 1, 2, 3):

            target_index = index + offset

            if not (
                0
                <= target_index
                < len(lines)
            ):
                continue

            target_text = _text(
                lines[target_index]
            )

            if forbidden_pattern.search(target_text):
                continue

            if cov_pattern.search(target_text):
                continue

            if _line_has_cov_context(target_index):
                continue

            date = _extract_date(target_text)

            if date and _is_plausible(date):
                return date

    # --------------------------------------------------------
    # 3. SPATIAL FALLBACK
    # --------------------------------------------------------

    for label_line in lines:

        label_text = _text(label_line)

        if not dob_pattern.search(label_text):
            continue

        label_bounds = _bounds(label_line)

        if label_bounds is None:
            continue

        lx1, ly1, lx2, ly2 = label_bounds

        label_center_y = (
            ly1 + ly2
        ) / 2

        label_height = max(
            ly2 - ly1,
            1,
        )

        candidates = []

        for candidate_line in lines:

            if candidate_line is label_line:
                continue

            candidate_text = _text(candidate_line)

            if forbidden_pattern.search(candidate_text):
                continue

            if cov_pattern.search(candidate_text):
                continue

            date = _extract_date(candidate_text)

            if not date:
                continue

            if not _is_plausible(date):
                continue

            bounds = _bounds(candidate_line)

            if bounds is None:
                continue

            cx1, cy1, cx2, cy2 = bounds

            candidate_center_y = (
                cy1 + cy2
            ) / 2

            vertical_distance = abs(
                candidate_center_y
                - label_center_y
            )

            if vertical_distance > (
                label_height * 4.0
            ):
                continue

            horizontal_distance = abs(
                cx1 - lx2
            )

            penalty = 0

            if cx1 < lx1:
                penalty += 500

            score = (
                vertical_distance
                + horizontal_distance * 0.05
                + penalty
            )

            candidates.append(
                (
                    score,
                    -_confidence(candidate_line),
                    date,
                )
            )

        if candidates:

            candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                )
            )

            return candidates[0][2]

    return None


# ============================================================
# DATE OF ISSUE
# ============================================================

def _find_issue_date(lines):
    """
    Extract Driving Licence Date of Issue.

    Handles common OCR variants of the DOI label:
        DOI, DO1, D0I, D01, Date of Issue

    Only the label is treated tolerantly; the date itself is
    extracted from OCR text and is never guessed.
    """

    candidates = []

    doi_pattern = re.compile(
        r"\bD[O0][I1]\b",
        flags=re.IGNORECASE,
    )

    date_of_issue_pattern = re.compile(
        r"\bDATE\s+OF\s+ISSUE\b",
        flags=re.IGNORECASE,
    )

    for index, line in enumerate(lines):

        text = _text(line)

        if not (
            doi_pattern.search(text)
            or date_of_issue_pattern.search(text)
        ):
            continue

        date = _extract_date(text)

        if date:
            candidates.append((index, date, _confidence(line), 0))
            continue

        if index + 1 < len(lines):
            next_line = lines[index + 1]
            next_date = _extract_date(_text(next_line))

            if next_date:
                candidates.append(
                    (index, next_date, _confidence(next_line), 1)
                )

        if index > 0:
            previous_line = lines[index - 1]
            previous_date = _extract_date(_text(previous_line))

            if previous_date:
                candidates.append(
                    (index, previous_date, _confidence(previous_line), 2)
                )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[0],
            item[3],
            -item[2],
        )
    )

    return candidates[0][1]


# ============================================================
# VALIDITY
# ============================================================

def _find_validity(lines):

    nt = None
    tr = None
    generic = None

    keywords = [
        "valid till",
        "validity",
        "valid upto",
        "valid up to",
    ]

    for index, line in enumerate(lines):

        text = _text(line)
        lower = text.lower()

        if not any(
            keyword in lower
            for keyword in keywords
        ):
            continue

        date = _extract_date(text)

        if (
            not date
            and index + 1 < len(lines)
        ):

            date = _extract_date(
                _text(
                    lines[index + 1]
                )
            )

        if not date:
            continue

        upper = text.upper()

        if re.search(
            r"\(\s*NT\s*\)|\bNT\b",
            upper,
        ):

            nt = date

        elif re.search(
            r"\(\s*TR\s*\)|\bTR\b",
            upper,
        ):

            tr = date

        else:

            generic = date

    if (
        nt is None
        and tr is None
        and generic
    ):
        nt = generic

    return nt, tr


# ============================================================
# BLOOD GROUP
# ============================================================

def _normalise_blood_group(text):

    if not text:
        return None

    value = str(text).upper()

    match = re.search(
        r"\b(AB|A|B|O)\s*([+-])",
        value,
    )

    if not match:
        return None

    candidate = (
        match.group(1)
        + match.group(2)
    )

    if candidate in VALID_BLOOD_GROUPS:
        return candidate

    return None


def _find_blood_group(lines):

    # Prefer explicitly labelled BG line.
    for index, line in enumerate(lines):

        text = _text(line)
        lower = text.lower()

        if (
            re.search(
                r"\bbg\b",
                lower,
            )
            or "blood group" in lower
        ):

            group = _normalise_blood_group(
                text
            )

            if group:
                return group

            # Check neighbouring lines.
            for offset in [1, -1]:

                target = index + offset

                if 0 <= target < len(lines):

                    group = (
                        _normalise_blood_group(
                            _text(
                                lines[target]
                            )
                        )
                    )

                    if group:
                        return group

    # Conservative fallback.
    for line in lines:

        group = _normalise_blood_group(
            _text(line)
        )

        if group:
            return group

    return None


# ============================================================
# CLASS OF VEHICLE
# ============================================================

def _find_cov(lines):
    """
    Find ALL vehicle classes.

    Handles both:

        MCWG

    and:

        MCWG04-10-2022

    and:

        COV: LMV MCWG
    """

    found = []

    # Longest first prevents shorter codes from
    # interfering with longer codes.
    codes = sorted(
        VALID_COV_CODES,
        key=len,
        reverse=True,
    )

    for line in lines:

        upper = _text(
            line
        ).upper()

        # Remove spaces for compact OCR output.
        compact = re.sub(
            r"\s+",
            "",
            upper,
        )

        for code in codes:

            compact_code = re.sub(
                r"\s+",
                "",
                code.upper(),
            )

            # ------------------------------------------------
            # EXACT MATCH
            # ------------------------------------------------

            if compact == compact_code:

                if code not in found:
                    found.append(code)

                break

            # ------------------------------------------------
            # CODE FOLLOWED BY DATE / PUNCTUATION
            #
            # Example:
            #
            # MCWG04-10-2022
            # LMV:04-10-2022
            # ------------------------------------------------

            pattern = (
                r"(?<![A-Z])"
                + re.escape(
                    compact_code
                )
                + r"(?="
                + r"\d{1,2}[-/.]"
                + r"|\s"
                + r"|[:;,()/\-]"
                + r"|$"
                + r")"
            )

            if re.search(
                pattern,
                compact,
            ):

                if code not in found:
                    found.append(code)

    return found


# ============================================================
# RELATIONSHIP DETECTION
# ============================================================

def _is_relationship_line(text):

    if not text:
        return False

    return bool(
        RELATIONSHIP_PATTERN.search(
            str(text)
        )
    )


# ============================================================
# NAME CLEANING
# ============================================================

def _clean_name(text):

    if not text:
        return None

    value = str(text).strip()

    # Remove Name label.
    value = re.sub(
        r"^\s*NAME\s*[:\-]?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    # Remove leading punctuation:
    #
    # :RISHABHAJITSINGH
    # -> RISHABHAJITSINGH

    value = re.sub(
        r"^[\s:;,\-.]+",
        "",
        value,
    )

    # Paddle can occasionally leave tiny fragments from labels.
    #
    # Example:
    # e :RISHABH AJIT SINGH

    value = re.sub(
        r"^\s*[A-Za-z]{1,2}\s*:\s*",
        "",
        value,
    )

    # Keep characters that can reasonably occur in a name.
    value = re.sub(
        r"[^A-Za-z .'-]",
        " ",
        value,
    )

    value = _clean_spaces(
        value
    )

    if len(value) < 3:
        return None

    return value


def _looks_like_person_name(value):

    if not value:
        return False

    if _is_relationship_line(
        value
    ):
        return False

    words = value.split()

    if not 1 <= len(words) <= 6:
        return False

    rejected_words = {
        "name",
        "dob",
        "doi",
        "do1",
        "d0i",
        "d01",
        "valid",
        "validity",
        "blood",
        "group",
        "address",
        "add",
        "signature",
        "thumb",
        "authority",
        "licence",
        "license",
        "india",
        "maharashtra",
        "vehicle",
        "vehicles",
        "form",
        "rule",
        "cov",
        "lmv",
        "mcwg",
        "mcwog",
        "issuing",
        "holder",
    }

    lowered = {
        word.lower()
        for word in words
    }

    if lowered & rejected_words:
        return False

    letters = sum(
        char.isalpha()
        for char in value
    )

    if (
        letters
        / max(
            len(value),
            1,
        )
        < 0.75
    ):
        return False

    # Reject dates/numbers masquerading as names.
    if re.search(
        r"\d",
        value,
    ):
        return False

    return True


# ============================================================
# NAME EXTRACTION
# ============================================================

def _find_name(lines):
    """
    Extract holder name around the Name label.

    Supports:

        Name : RISHABH AJIT SINGH

    and:

        Name
        RISHABH AJIT SINGH

    and Paddle ordering:

        :RISHABHAJITSINGH
        Name
        S/D/WofAJIT SINGH

    Relationship-labelled lines are explicitly rejected.
    """

    for index, line in enumerate(lines):

        text = _text(line)

        if not re.search(
            r"\bname\b",
            text,
            flags=re.IGNORECASE,
        ):
            continue

        # ====================================================
        # SAME LINE
        # ====================================================

        same_line = re.sub(
            r".*?\bname\b\s*[:\-]?",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        if (
            same_line
            and not _is_relationship_line(
                same_line
            )
        ):

            candidate = _clean_name(
                same_line
            )

            if _looks_like_person_name(
                candidate
            ):
                return candidate

        # ====================================================
        # NEIGHBOURING LINES
        #
        # Search closest lines first.
        #
        # Important:
        # Paddle may return the value BEFORE the visual Name label.
        # ====================================================

        search_offsets = [
            -1,
            1,
            -2,
            2,
        ]

        for offset in search_offsets:

            target_index = (
                index
                + offset
            )

            if not (
                0
                <= target_index
                < len(lines)
            ):
                continue

            raw_candidate = _text(
                lines[target_index]
            )

            # Never treat relationship line as holder name.
            if _is_relationship_line(
                raw_candidate
            ):
                continue

            # Reject obvious labels.
            if re.search(
                r"\b("
                r"DOB"
                r"|D[O0][I1]"
                r"|BG"
                r"|COV"
                r"|ADD"
                r"|ADDRESS"
                r"|PIN"
                r"|VALID"
                r")\b",
                raw_candidate,
                flags=re.IGNORECASE,
            ):
                continue

            candidate = _clean_name(
                raw_candidate
            )

            if _looks_like_person_name(
                candidate
            ):
                return candidate

    return None


# ============================================================
# RELATIVE NAME
# ============================================================

def _clean_relative_name(text):

    if not text:
        return None

    value = str(text)

    patterns = [
        r"^\s*S\s*/\s*D\s*/\s*M?\s*W\s*OF\s*",
        r"^\s*S\s*/\s*D\s*/\s*W\s*OF\s*",
        r"^\s*S\s*/\s*D\s*/\s*H\s*OF\s*",
        r"^\s*S\s*/\s*O\s*",
        r"^\s*D\s*/\s*O\s*",
        r"^\s*W\s*/\s*O\s*",
        r"^\s*C\s*/\s*O\s*",
        r"^\s*SON\s+OF\s*",
        r"^\s*DAUGHTER\s+OF\s*",
        r"^\s*WIFE\s+OF\s*",
    ]

    for pattern in patterns:

        value = re.sub(
            pattern,
            "",
            value,
            flags=re.IGNORECASE,
        )

    value = re.sub(
        r"[^A-Za-z .'-]",
        " ",
        value,
    )

    value = _clean_spaces(
        value
    )

    if len(value) < 3:
        return None

    return value


def _find_relative_name(lines):

    for index, line in enumerate(lines):

        text = _text(line)

        if not _is_relationship_line(
            text
        ):
            continue

        candidate = (
            _clean_relative_name(
                text
            )
        )

        if _looks_like_person_name(
            candidate
        ):
            return candidate

        if index + 1 < len(lines):

            candidate = _clean_name(
                _text(
                    lines[index + 1]
                )
            )

            if _looks_like_person_name(
                candidate
            ):
                return candidate

    return None


# ============================================================
# HOLDER NAME SPACING REPAIR
# ============================================================

def _repair_name_spacing(name, relative_name=None):
    """
    Repair missing spaces in the holder name using independently
    extracted relative-name evidence.

    Example:
        holder OCR:   RISHABHAJIT SINGH
        relative OCR: AJIT SINGH
        repaired:     RISHABH AJIT SINGH

    No names are hardcoded. A repair is made only when the compact
    relative name is an exact suffix of the compact holder name, and
    the repair changes whitespace only.
    """

    if not name:
        return None

    name = _clean_spaces(name)

    if not relative_name:
        return name

    relative_name = _clean_spaces(relative_name)

    compact_name = re.sub(
        r"[^A-Za-z]",
        "",
        name,
    ).upper()

    compact_relative = re.sub(
        r"[^A-Za-z]",
        "",
        relative_name,
    ).upper()

    if not compact_name or not compact_relative:
        return name

    if len(compact_relative) >= len(compact_name):
        return name

    if not compact_name.endswith(compact_relative):
        return name

    prefix_length = len(compact_name) - len(compact_relative)

    if prefix_length < 2:
        return name

    holder_prefix = compact_name[:prefix_length]

    repaired_name = _clean_spaces(
        holder_prefix + " " + relative_name
    )

    repaired_compact = re.sub(
        r"[^A-Za-z]",
        "",
        repaired_name,
    ).upper()

    if repaired_compact != compact_name:
        return name

    return repaired_name


# ============================================================
# PIN CODE
# ============================================================

def _find_pin_code(lines):

    # Prefer labelled PIN.
    for line in lines:

        text = _text(line)

        if "pin" not in text.lower():
            continue

        match = PIN_PATTERN.search(
            text
        )

        if match:
            return match.group(1)

    # Fallback.
    for line in lines:

        match = PIN_PATTERN.search(
            _text(line)
        )

        if match:
            return match.group(1)

    return None


# ============================================================
# ADDRESS
# ============================================================

def _clean_address_start(text):

    value = str(text)

    value = re.sub(
        r"^\s*(?:ADD|ADDRESS)\s*[:\-]?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    return _clean_spaces(
        value
    )


def _find_address(lines):
    """
    Extract address from the DL while rejecting OCR contamination
    from signatures, photographs, stamps, etc.

    Strategy:
        1. Find Add / Address label.
        2. Determine the address column from its bbox.
        3. Collect subsequent lines from approximately the same column.
        4. Stop at PIN / Signature / Issuing Authority.
        5. Ignore detections far to the right.

    Example OCR:

        Add:BLOCK NO-F/5, R.NO.3,       x=85
        DEONAR NEW MUNICIPAL COLONY,    x=84
        Greater Mumbai,...,MH           x=82

        SKamb                            x=1151   <-- rejected

        PIN:400043                       x=85
    """

    # ========================================================
    # FIND ADDRESS START
    # ========================================================

    address_start = None
    address_bounds = None

    address_pattern = re.compile(
        r"^\s*(?:ADD|ADDRESS)\s*[:\-]?",
        flags=re.IGNORECASE,
    )

    for index, line in enumerate(lines):

        text = _text(line)

        if address_pattern.search(text):

            address_start = index
            address_bounds = _bounds(line)

            break

    if address_start is None:
        return None

    # ========================================================
    # ADDRESS COLUMN INFORMATION
    # ========================================================

    address_left = None
    address_right = None

    if address_bounds is not None:

        address_left = address_bounds[0]
        address_right = address_bounds[2]

    # ========================================================
    # FIRST ADDRESS LINE
    # ========================================================

    first_text = _text(
        lines[address_start]
    )

    # Remove Add / Address label.

    first_text = address_pattern.sub(
        "",
        first_text,
        count=1,
    )

    first_text = _clean_spaces(
        first_text
    )

    parts = []

    if first_text:
        parts.append(first_text)

    # ========================================================
    # HARD STOP PATTERNS
    # ========================================================

    stop_pattern = re.compile(
        r"(?:"
        r"\bPIN\b"
        r"|SIGNATURE"
        r"|ISSUING\s*AUTHORITY"
        r"|ISSUINGAUTHORITY"
        r"|IMPRESSION\s+OF\s+HOLDER"
        r"|THUMB\s*IMPRESSION"
        r")",
        flags=re.IGNORECASE,
    )

    # ========================================================
    # OTHER FIELD LABELS
    # ========================================================

    field_pattern = re.compile(
        r"^\s*(?:"
        r"DOB"
        r"|D[O0][I1]"
        r"|BG"
        r"|COV"
        r"|NAME"
        r"|S/D/W"
        r"|S/O"
        r"|D/O"
        r"|W/O"
        r"|VALID"
        r"|VALIDITY"
        r"|DL\s*NO"
        r")\b",
        flags=re.IGNORECASE,
    )

    # ========================================================
    # COLLECT FOLLOWING ADDRESS LINES
    # ========================================================

    for index in range(
        address_start + 1,
        len(lines),
    ):

        line = lines[index]

        text = _clean_spaces(
            _text(line)
        )

        if not text:
            continue

        # ----------------------------------------------------
        # HARD STOP
        # ----------------------------------------------------

        if stop_pattern.search(text):
            break

        # ----------------------------------------------------
        # OTHER STRUCTURED FIELD
        # ----------------------------------------------------

        if field_pattern.search(text):
            break

        # ----------------------------------------------------
        # BOUNDING BOX FILTER
        # ----------------------------------------------------

        bounds = _bounds(line)

        if (
            bounds is not None
            and address_left is not None
        ):

            x1, y1, x2, y2 = bounds

            # Address starts around x=85 on your test DL.
            #
            # Genuine continuation lines:
            #
            #     x=84
            #     x=82
            #
            # Signature OCR "SKamb":
            #
            #     x=1151
            #
            # Therefore anything starting far away from the
            # address column should not be concatenated.

            max_left_shift = 100
            max_right_shift = 600

            if x1 < (
                address_left
                - max_left_shift
            ):
                continue

            if x1 > (
                address_left
                + max_right_shift
            ):
                continue

        # ----------------------------------------------------
        # ADD VALID ADDRESS LINE
        # ----------------------------------------------------

        parts.append(text)

        # Defensive limit. Indian DL addresses normally don't
        # require an unlimited number of OCR lines.

        if len(parts) >= 6:
            break

    # ========================================================
    # FINAL ADDRESS
    # ========================================================

    if not parts:
        return None

    address = " ".join(parts)

    # Normalize whitespace.

    address = _clean_spaces(
        address
    )

    # Remove spaces before commas.

    address = re.sub(
        r"\s+,",
        ",",
        address,
    )

    # Normalize repeated commas.

    address = re.sub(
        r",\s*,+",
        ",",
        address,
    )

    return address.strip(
        " ,"
    )


# ============================================================
# RAW TEXT
# ============================================================

def _full_text(lines):

    return "\n".join(
        _text(line)
        for line in lines
        if _text(line)
    )


# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================

def extract_fields(
    ocr_lines,
    min_confidence=0.0,
):
    """
    Convert PaddleOCR/EasyOCR output into structured DL fields.
    """

    # ========================================================
    # FILTER CONFIDENCE
    # ========================================================

    lines = [
        line
        for line in ocr_lines
        if _confidence(line)
        >= min_confidence
    ]

    # ========================================================
    # SORT
    # ========================================================

    lines = _sort_lines(
        lines
    )

    # ========================================================
    # DL NUMBER
    # ========================================================

    (
        dl_number,
        other_dl_candidates,
    ) = _find_dl_number(
        lines
    )

    # ========================================================
    # DATES  --  DOI first so it can constrain DOB search
    # ========================================================

    date_of_issue = _find_issue_date(
        lines
    )

    date_of_birth = _find_dob(
        lines,
        max_date=date_of_issue,
    )

    (
        validity_non_transport,
        validity_transport,
    ) = _find_validity(
        lines
    )

    # ========================================================
    # HARD GUARD: DOB must be strictly before DOI
    # ========================================================

    if date_of_birth and date_of_issue:
        dob_t = _date_tuple(date_of_birth)
        doi_t = _date_tuple(date_of_issue)
        if dob_t and doi_t and dob_t >= doi_t:
            date_of_birth = None

    # ========================================================
    # PERSONAL DETAILS
    # ========================================================

    relative_name = _find_relative_name(
        lines
    )

    name = _find_name(
        lines
    )

    name = _repair_name_spacing(
        name,
        relative_name,
    )

    blood_group = _find_blood_group(
        lines
    )

    # ========================================================
    # COV
    # ========================================================

    class_of_vehicle = _find_cov(
        lines
    )

    # ========================================================
    # ADDRESS
    # ========================================================

    address = _find_address(
        lines
    )

    pin_code = _find_pin_code(
        lines
    )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "dl_number":
            dl_number,

        "dl_number_other_candidates":
            other_dl_candidates,

        "name":
            name,

        "relative_name":
            relative_name,

        "date_of_birth":
            date_of_birth,

        "date_of_issue":
            date_of_issue,

        "validity_non_transport":
            validity_non_transport,

        "validity_transport":
            validity_transport,

        "blood_group":
            blood_group,

        "class_of_vehicle":
            class_of_vehicle,

        "address":
            address,

        "pin_code":
            pin_code,

        "raw_text":
            _full_text(
                lines
            ),
    }
