"""TransUnion CIBIL document verification core.

Converted from the supplied cibil_verification.ipynb.
The notebook's verification logic is preserved; notebook installation,
quick-test, and standalone FastAPI cells are intentionally excluded."""

import json
import os
import re
from datetime import datetime, date
from pathlib import Path

import fitz
import pdfplumber


# ============================================================
# SOURCE NOTEBOOK CELL 3
# ============================================================

# Notebook-only imports removed; required imports are declared above.


# ============================================================
# SOURCE NOTEBOOK CELL 5
# ============================================================

def check_file(file_bytes):
    """Fast PDF security/structural validation using PDF bytes."""
    result = {
        "passed": False,
        "reason": "",
        "pages": 0,
        "chars": 0,
        "native_text": False,
    }

    try:
        if not file_bytes or not file_bytes.startswith(b"%PDF"):
            result["reason"] = "Not a valid PDF file"
            return result

        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            pages = len(pdf)
            if pages == 0:
                result["reason"] = "PDF has 0 pages"
                return result

            # Read only the first few pages for the fast classification path.
            # The full text is extracted once later only when needed.
            sample_chars = 0
            sample_pages = min(pages, 6)
            for i in range(sample_pages):
                try:
                    sample_chars += len(pdf[i].get_text("text") or "")
                except Exception:
                    continue

            result.update({
                "passed": True,
                "pages": pages,
                "chars": sample_chars,
                "native_text": sample_chars >= 100,
            })

    except Exception as e:
        result["reason"] = f"Cannot open PDF: {e}"

    return result


def extract_full_text(file_bytes):
    """Extract native PDF text once. OCR is intentionally not used."""
    full = []

    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page in pdf:
            try:
                text = page.get_text("text") or ""
            except Exception:
                text = ""
            if text:
                full.append(text)

    return "\n".join(full)


def detect_document_type(text):
    """
    Strong multi-document classifier for the CIBIL endpoint.

    Known classes:
        CIBIL, CRIF, BANK_STATEMENT, SALARY_SLIP, ITR, UNKNOWN

    Generic words such as CREDIT, TAX, PAN or ACCOUNT are never enough
    by themselves to classify a document. CIBIL requires CIBIL-specific
    identity evidence plus supporting credit-report structure.
    """

    text_upper = re.sub(r"\s+", " ", (text or "").upper())

    signatures = {
        "CIBIL": {
            "strong": (
                "TRANSUNION CIBIL",
                "CIBILTUSC3",
                "CONSUMER CIR",
                "CIBIL TRANSUNION SCORE",
                "TRANSUNION CIBIL LIMITED",
                "CIBIL TRANSUNION",
            ),
            "medium": (
                "CIBIL SCORE",
                "CREDIT INFORMATION REPORT",
                "CREDIT REPORT",
                "CREDIT HISTORY",
                "PAYMENT HISTORY",
                "DAYS PAST DUE",
                "DPD",
                "ACCOUNT STATUS",
                "ENQUIRIES",
                "ENQUIRY",
                "CONTROL NUMBER",
                "MEMBER REFERENCE NUMBER",
                "CREDIT ACCOUNT",
                "REPAYMENT HISTORY",
            ),
        },
        "CRIF": {
            "strong": (
                "CRIF HIGH MARK",
                "CRIF HM SCORE",
                "PERFORM CONSUMER",
                "CREDIT INFORMATION REPORT PROV2",
                "CHM REF #",
            ),
            "medium": (
                "CRIF SCORE",
                "CRIF HIGHMARK",
                "CREDIT REPORT",
                "DAYS PAST DUE",
                "ENQUIRIES",
                "PAYMENT HISTORY",
            ),
        },
        "BANK_STATEMENT": {
            "strong": (
                "BANK STATEMENT",
                "ACCOUNT STATEMENT",
                "STATEMENT OF ACCOUNT",
                "TRANSACTION STATEMENT",
            ),
            "medium": (
                "OPENING BALANCE",
                "CLOSING BALANCE",
                "WITHDRAWAL",
                "DEPOSIT",
                "TRANSACTION DATE",
                "VALUE DATE",
                "IFSC",
                "NEFT",
                "RTGS",
                "IMPS",
            ),
        },
        "SALARY_SLIP": {
            "strong": (
                "PAYSLIP",
                "PAY SLIP",
                "SALARY SLIP",
                "SALARY STATEMENT",
                "MONTHLY PAYSLIP",
            ),
            "medium": (
                "SALARY DETAILS",
                "EARNINGS",
                "TOTAL EARNINGS",
                "DEDUCTIONS",
                "TAXES & DEDUCTIONS",
                "NET SALARY",
                "NET SALARY PAYABLE",
                "EMPLOYEE ID",
                "EMPLOYEE NUMBER",
                "BASIC SALARY",
                "GROSS SALARY",
                "PROVIDENT FUND",
                "PF EMPLOYEE",
            ),
        },
        "ITR": {
            "strong": (
                "INCOME TAX RETURN",
                "FORM ITR",
                "ITR-V",
                "RETURN OF INCOME",
                "E-FILING ACKNOWLEDGEMENT",
            ),
            "medium": (
                "ASSESSMENT YEAR",
                "TAXABLE INCOME",
                "TOTAL INCOME",
                "TAX PAYABLE",
                "COMPUTATION OF INCOME",
                "CAPITAL GAINS",
                "INCOME FROM HOUSE PROPERTY",
                "INCOME FROM OTHER SOURCES",
            ),
        },
    }

    scores = {}
    evidence = {}

    for name, sig in signatures.items():
        strong = [x for x in sig["strong"] if x in text_upper]
        medium = [x for x in sig["medium"] if x in text_upper]

        # Strong identity evidence dominates. Medium terms provide support.
        score = min(70, len(strong) * 35) + min(30, len(medium) * 5)

        scores[name] = score
        evidence[name] = {
            "strong_hits": strong,
            "medium_hits": medium,
        }

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best, best_score = ranking[0]
    second_score = ranking[1][1] if len(ranking) > 1 else 0
    margin = best_score - second_score

    cibil = evidence["CIBIL"]
    cibil_strong = len(cibil["strong_hits"])
    cibil_medium = len(cibil["medium_hits"])

    cibil_confident = bool(
        cibil_strong >= 1
        and (
            cibil_strong >= 2
            or cibil_medium >= 3
        )
        and (
            best == "CIBIL"
            and (margin >= 10 or best_score >= 90)
        )
    )

    # A strong competing document must win immediately.
    if best != "CIBIL" and best_score >= 70 and margin >= 15:
        detected = best
    elif best == "CIBIL" and cibil_confident:
        detected = "CIBIL"
    elif best != "CIBIL" and best_score >= 50 and margin >= 10:
        detected = best
    else:
        detected = "UNKNOWN"

    return {
        "detected": detected,
        "confidence": (
            "HIGH" if detected != "UNKNOWN" and best_score >= 85
            else "MEDIUM" if detected != "UNKNOWN"
            else "NONE"
        ),
        "scores": scores,
        "ranking": ranking,
        "margin": margin,
        "evidence": evidence,
        "reason": (
            f"Detected as {detected} with score {best_score}"
            if detected != "UNKNOWN"
            else "Document classification is ambiguous"
        ),
    }


# ============================================================
# SOURCE NOTEBOOK CELL 7
# ============================================================

CIBIL_SIGNATURES = [
    "TRANSUNION CIBIL",
    "CIBILTUSC3",
    "CONSUMER CIR",
    "TRANSUNION CIBIL LIMITED",
    "CIBIL TRANSUNION SCORE",
]

def detect_bureau(text):
    """Return dict with is_cibil flag, confidence level, and matched keywords."""
    text_upper = text.upper()
    matched = [k for k in CIBIL_SIGNATURES if k in text_upper]

    if len(matched) >= 2:
        return {"is_cibil": True, "confidence": "HIGH", "matched": matched}

    if len(matched) == 1:
        return {
            "is_cibil": True,
            "confidence": "LOW",
            "matched": matched,
            "reason": "Only 1 CIBIL keyword matched — may be partial or tampered document",
        }

    return {
        "is_cibil": False,
        "reason": "No CIBIL keywords found — this is not a TransUnion CIBIL report",
    }


# ============================================================
# SOURCE NOTEBOOK CELL 9
# ============================================================

def extract_header_info(text):
    """Extract all applicant-level metadata from the CIBIL report header."""
    info = {}

    # Report date — "DATE: 18-04-2026" (appears in header line)
    m = re.search(r"DATE:\s*(\d{2}-\d{2}-\d{4})", text)
    if m:
        info["date_of_report_raw"] = m.group(1)
        try:
            info["date_of_report"] = datetime.strptime(m.group(1), "%d-%m-%Y").date()
        except ValueError:
            pass

    # Control number (CIBIL internal report ID)
    m = re.search(r"CONTROL NUMBER:\s*(\d+)", text)
    if m:
        info["control_number"] = m.group(1)

    # Member reference number
    m = re.search(r"MEMBER REFERENCE NUMBER:\s*(\d+)", text)
    if m:
        info["member_ref"] = m.group(1)

    # Applicant name — "NAME: SHAIKH SALIM SHAIKH IBRAHIM BAGBAN"
    m = re.search(r'NAME:\s*([A-Z][A-Z\s]+?)\n', text)
    if m:
        info["name"] = m.group(1).strip()

    # Date of birth
    m = re.search(r"DATE OF BIRTH:\s*(\d{2}-\d{2}-\d{4})", text)
    if m:
        info["dob"] = m.group(1)

    # Gender — Ghulam Ali has GENDER: 3 (data entry error) — handle gracefully
    m = re.search(r"GENDER:\s*(\w+)", text)
    if m:
        g = m.group(1)
        info["gender"] = g if g in ("MALE", "FEMALE") else f"UNKNOWN({g})"

    # PAN — handles both normal and (e) enquiry-provided variants
    # e.g. "INCOME TAX ID NUMBER (PAN)  AJOPB4590P"
    # e.g. "INCOME TAX ID NUMBER (PAN) (e)  NQOPK8316Q"
    # pans = re.findall(
    #     r"INCOME TAX ID NUMBER \(PAN\)(?:\s*\(e\))?\s+([A-Z]{5}[0-9]{4}[A-Z])",
    #     text.upper(),
    # )

    pans = re.findall(
    r"INCOME\s+TAX\s+ID\s+NUMBER\s+\(PAN\)\s*(?:\(\s*E\s*\))?\s*([A-Z]{5}[0-9]{4}[A-Z])",
    text.upper(),
    re.IGNORECASE,
)
    unique_pans = list(set(pans))
    info["pans_found"] = unique_pans
    info["pan"] = unique_pans[0] if len(unique_pans) == 1 else None
    info["multiple_pans"] = len(unique_pans) > 1

    # Income — "124055 GROSS INCOME MONTHLY" or "124055 NET INCOME ANNUAL"
    m = re.search(r'\b(\d{4,7})\s+(GROSS|NET)\s+INCOME\s+(MONTHLY|ANNUAL)', text)
    if m:
        income = int(m.group(1))
        freq = m.group(3)
        info["income_monthly"] = income if freq == "MONTHLY" else round(income / 12)
        info["income_type"] = m.group(2)
        info["income_freq"] = freq

    return info


# ============================================================
# SOURCE NOTEBOOK CELL 11
# ============================================================

def extract_score(text):
    """Extract CIBIL score. Handles normal (300-900), -1, 0, and 1-5 special cases."""
    m = re.search(r"CIBILTUSC3\s+(-?\d+)", text)
    if not m:
        return {"found": False, "score": None, "reason": "CIBIL score not found in document"}

    raw = int(m.group(1))

    if raw == -1:
        return {
            "found": True, "score": -1, "category": "NOT_IN_CIBIL",
            "reason": "Score -1: Consumer not in CIBIL database or insufficient information for scoring",
        }

    if raw == 0:
        return {
            "found": True, "score": 0, "category": "SCORING_ERROR",
            "reason": "Score 0: CIBIL scoring error — manual review required",
        }

    if 1 <= raw <= 5:
        return {
            "found": True, "score": raw, "category": "NEW_TO_CREDIT",
            "reason": (
                f"Score {raw} (1-5 scale): Less than 6 months credit history — "
                "new-to-credit applicant"
            ),
        }

    if 300 <= raw <= 900:
        return {"found": True, "score": raw, "source": "CIBILTUSC3"}

    # Out of valid range — possible tampering
    return {
        "found": True, "score": raw, "category": "INVALID_RANGE",
        "reason": f"Score {raw} is outside valid CIBIL range (300-900) — possible tampered document",
    }


def categorize_score(score):
    """Return (category_label, risk_level) for a normal CIBIL score 300-900."""
    if score >= 800: return "EXCELLENT",    "VERY_LOW"
    if score >= 750: return "GOOD",         "LOW"
    if score >= 700: return "ABOVE_AVERAGE","LOW_MEDIUM"
    if score >= 650: return "AVERAGE",      "MEDIUM"
    if score >= 550: return "BELOW_AVERAGE","HIGH"
    return "POOR", "VERY_HIGH"


def extract_scoring_factors(text):
    """
    CIBIL lists scoring factors explaining WHY the score is what it is.
    They appear right after the CIBILTUSC3 score number, before POSSIBLE RANGE.
    e.g.  1: HIGH BALANCE BUILD-UP ON NON-MORTGAGE LOANS
          2: HIGH PROPORTION OF OUTSTANDING TRADES
    Max 5 factors.
    """
    m = re.search(r'CIBILTUSC3\s+[-\d]+\s*\n([\s\S]+?)POSSIBLE RANGE', text)
    if not m: return []
    block = m.group(1)
    # Each factor is "N: SOME TEXT" on its own line
    factors = re.findall(r'(?:^|\n)\s*\d+[:\s]+([A-Z][A-Z\s/(),-]{4,70}?)(?=\n\s*\d|\nPOSSIBLE|$)', block)
    return [f.strip() for f in factors if len(f.strip()) > 4][:5]

# ============================================================
# SOURCE NOTEBOOK CELL 13
# ============================================================

def extract_account_summary(text):
    info = {}
 
    m = re.search(
        r'All Accounts TOTAL:\s*(\d+)\s+HIGH CR/SANC\.\s*AMT:\s*([\d,]+)\s+CURRENT:\s*([\d,]+)',
        text
    )
    if m:
        info["total_accounts"]        = int(m.group(1))
        info["total_sanctioned"]      = int(m.group(2).replace(',', ''))
        info["total_current_balance"] = int(m.group(3).replace(',', ''))
 
    # Two OVERDUE fields: count then amount
    m = re.search(r'OVERDUE:\s*(\d+)\s+OVERDUE:\s*([\d,]+)', text)
    if m:
        info["overdue_accounts"] = int(m.group(1))
        info["overdue_amount"]   = int(m.group(2).replace(',', ''))
 
    m = re.search(r'ZERO-BALANCE:\s*(\d+)', text)
    if m:
        info["zero_balance_accounts"] = int(m.group(1))
 
    m = re.search(r'RECENT:\s*(\d{2}-\d{2}-\d{4})', text)
    if m:
        info["most_recent_account"] = m.group(1)
 
    m = re.search(r'OLDEST:\s*(\d{2}-\d{2}-\d{4})', text)
    if m:
        info["oldest_account"] = m.group(1)
 
    return info

# ============================================================
# SOURCE NOTEBOOK CELL 15
# ============================================================

def dpd_to_severity(dpd_val):
    """Map a single DPD integer to severity 0-4."""
    if dpd_val == 900: return 4   # Written off / Loss asset
    if dpd_val >= 180: return 4   # Loss
    if dpd_val >= 90:  return 3   # Doubtful
    if dpd_val >= 30:  return 2   # Sub-standard
    if dpd_val >= 1:   return 1   # Special Mention
    return 0                       # Standard


SEV_LABEL = {0: "STD", 1: "SMA", 2: "SUB", 3: "DBT", 4: "LOS"}


def parse_cibil_dpd(block):
    """
    Parse CIBIL payment history from an account block.

    CIBIL DPD format:
        DAYS PAST DUE/ASSET CLASSIFICATION (UP TO 36 MONTHS; LEFT TO RIGHT)
        000  022  000  162  000
        03-26  02-26  01-26  12-25  11-25

    Returns:
        worst_dpd      : highest DPD value seen (900 if written off)
        worst_severity : 0-4 scale
        worst_label    : STD / SMA / SUB / DBT / LOS
        bad_months     : count of months with any DPD > 0
        has_writeoff   : True if 900 was found (written-off indicator)
        dpd_values     : list of all DPD integers found
    """
    dpd_section = re.search(r"DAYS PAST DUE.*", block, re.DOTALL)
    if not dpd_section:
        return {
            "worst_dpd": 0, "worst_severity": 0, "worst_label": "STD",
            "bad_months": 0, "has_writeoff": False, "dpd_values": [],
        }

    section = dpd_section.group()

    # Extract 3-digit numbers that are NOT part of dates (dates look like "03-26")
    # Strategy: find \d{3} not immediately followed by a dash+2digits
    dpd_vals = re.findall(r'(?<![,\d])(\d{3})(?!\d)(?!\s*-\s*\d{2})', section)
 
    has_900 = '900' in dpd_vals
    ints    = [int(v) for v in dpd_vals if v != '900']
 
    worst = 900 if has_900 else (max(ints) if ints else 0)
    ws    = dpd_to_severity(worst)
    bad   = sum(1 for d in ints if d > 0) + (1 if has_900 else 0)
 
    return {
        "worst_dpd":      worst,
        "worst_severity": ws,
        "worst_label":    SEV_LABEL[ws],
        "bad_months":     bad,
        "has_writeoff":   has_900,
        "rbi_classification": get_rbi_classification(worst)
    }


def get_rbi_classification(worst_dpd):
    """
    RBI asset classification based on DPD.
    This is the legal classification lenders use for provisioning.
    """
    if worst_dpd == 0:    return "STANDARD"
    if worst_dpd < 30:    return "SPECIAL_MENTION"
    if worst_dpd < 90:    return "SUB_STANDARD"
    if worst_dpd < 180:   return "DOUBTFUL_1"
    if worst_dpd < 270:   return "DOUBTFUL_2"
    if worst_dpd < 360:   return "DOUBTFUL_3"
    if worst_dpd == 900:  return "LOSS"
    return "LOSS"
 

# ============================================================
# SOURCE NOTEBOOK CELL 17
# ============================================================

SECURED_TYPES = [
    "PROPERTY LOAN", "HOME LOAN", "HOUSING LOAN", "GOLD LOAN",
    "TWO-WHEELER LOAN", "TWO WHEELER", "VEHICLE LOAN", "CAR LOAN",
    "COMMERCIAL VEHICLE", "LOAN AGAINST", "TRACTOR LOAN", "MACHINERY LOAN"
]
 
OWNERSHIP_WEIGHT = {
    "INDIVIDUAL":      1.0,
    "JOIN":            0.6,   # CIBIL uses JOIN not JOINT
    "JOINT":           0.6,
    "GUARANTOR":       0.4,
    "AUTHORISED USER": 0.1,   # Credit card add-on — minimal liability
}
 
 
def parse_account(block, acc_num):
    acc = {"account_number": str(acc_num)}
 
    # TYPE — extract text up to REPORTED AND CERTIFIED or end-of-line
    # Real data: "TYPE: 61  REPORTED AND CERTIFIED: EMI: 89,001"
    # Real data: "TYPE: PERSONAL LOAN  REPORTED AND CERTIFIED:"
    m = re.search(r'TYPE:\s*([^\n]+?)(?:\s{2,}|\s+REPORTED)', block)
    if m:
        acc["account_type"] = m.group(1).strip()
 
    # Ownership — CIBIL uses JOIN not JOINT
    m = re.search(r'OWNERSHIP:\s*(\w+)', block)
    if m:
        acc["ownership"] = m.group(1).strip()
 
    # Status — closed only if CLOSED: date present
    cm = re.search(r'CLOSED:\s*(\d{2}-\d{2}-\d{4})', block)
    acc["status"]      = "CLOSED" if cm else "ACTIVE"
    acc["closed_date"] = cm.group(1) if cm else None
 
    m = re.search(r'OPENED:\s*(\d{2}-\d{2}-\d{4})', block)
    if m:
        acc["opened_date"] = m.group(1)
 
    # SANCTIONED — real CIBIL has NO space: "SANCTIONED:25,00,000"
    # Allow optional space to handle edge cases
    m = re.search(r'SANCTIONED:[ \t]*([\d,]+)', block)
    if m:
        acc["sanctioned_amount"] = int(m.group(1).replace(',', ''))
 
    m = re.search(r'CURRENT BALANCE:\s*([-\d,]+)', block)
    if m:
        try:
            acc["current_balance"] = int(m.group(1).replace(',', ''))
        except ValueError:
            acc["current_balance"] = 0
 
    m = re.search(r'\bEMI:\s*([\d,]+)', block)
    if m:
        acc["emi_amount"] = int(m.group(1).replace(',', ''))
 
    m = re.search(r'LAST PAYMENT:\s*(\d{2}-\d{2}-\d{4})', block)
    if m:
        acc["last_payment_date"] = m.group(1)
 
    # Last reported/certified date — used for delinquency recency
    m = re.search(r'REPORTED AND CERTIFIED:\s*(\d{2}-\d{2}-\d{4})', block)
    if m:
        acc["last_reported_date"] = m.group(1)
 
    m = re.search(r'REPAYMENT TENURE:\s*(\d+)', block)
    if m:
        acc["repayment_tenure"] = int(m.group(1))
 
    m = re.search(r'INTEREST RATE:\s*([\d.]+)', block)
    if m:
        acc["interest_rate"] = float(m.group(1))
 
    m = re.search(r'COLLATERAL VALUE:\s*([\d,]+)', block)
    if m:
        acc["collateral_value"] = int(m.group(1).replace(',', ''))
 
    m = re.search(r'COLLATERAL TYPE:\s*(.+?)(?:\s{2,}|\n|PMT)', block)
    if m:
        acc["collateral_type"] = m.group(1).strip()
 
    # ── Written-off / Settlement ──────────────────────────────
    acc["is_written_off"]    = False
    acc["is_settled"]        = False
    acc["total_writeoff"]    = 0
    acc["settlement_amount"] = 0
 
    if 'WRITTEN OFF' in block:
        m = re.search(r'WRITTEN OFF \(TOTAL\):\s*([\d,]+)', block)
        if m:
            acc["total_writeoff"] = int(m.group(1).replace(',', ''))
            acc["is_written_off"] = acc["total_writeoff"] > 0
 
        m = re.search(r'\bSETTLEMENT:\s*([\d,]+)', block)
        if m:
            acc["settlement_amount"] = int(m.group(1).replace(',', ''))
            acc["is_settled"]        = acc["settlement_amount"] > 0
 
        m = re.search(r'WRITTEN OFF /SETTLED STATUS:\s*\n(.+)', block)
        if m:
            acc["writeoff_status"] = m.group(1).strip()
 
    # ── Derived flags ─────────────────────────────────────────
    atype = acc.get("account_type", "").upper()
    acc["is_secured"]       = any(s in atype for s in SECURED_TYPES)
    acc["ownership_weight"] = OWNERSHIP_WEIGHT.get(
        acc.get("ownership", "INDIVIDUAL").upper(), 1.0
    )
 
    # ── DPD ───────────────────────────────────────────────────
    dpd = parse_cibil_dpd(block)
    acc["payment_history"] = dpd
 
    acc["has_issues"] = (
        acc["is_written_off"]     or
        acc["is_settled"]         or
        dpd["has_writeoff"]       or
        dpd["worst_severity"] >= 2
    )
    return acc
 
 
def extract_all_accounts(text):
    """
    CIBIL account separator: 'ACCOUNT  DATES  AMOUNTS  STATUS'
    First chunk is pre-accounts header — skip it.
    """
    blocks = re.split(r'ACCOUNT\s+DATES\s+AMOUNTS\s+STATUS', text)
    return [parse_account(b, i) for i, b in enumerate(blocks[1:], 1)]
 

# ============================================================
# SOURCE NOTEBOOK CELL 19
# ============================================================

def analyse_enquiries(text):
    """
    Parse CIBIL enquiry summary row.
    CIBIL pre-computes the counts so we use those directly.
    """
    # "All Enquiries  78  2  28  20  18-04-2026"
    m = re.search(
        r"All Enquiries\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
        text,
    )

    if not m:
        return {
            "total": 0, "past_30d": 0, "past_12m": 0, "past_24m": 0, "flags": [],
        }

    total   = int(m.group(1))
    past_30 = int(m.group(2))
    past_12 = int(m.group(3))
    past_24 = int(m.group(4))

    flags = []

    if past_30 >= 5:
        flags.append({
            "flag": "HIGH_RECENT_ENQUIRIES", "severity": "HIGH",
            "detail": f"{past_30} enquiries in last 30 days — very active loan shopping",
        })
    elif past_30 >= 3:
        flags.append({
            "flag": "MODERATE_RECENT_ENQUIRIES", "severity": "MEDIUM",
            "detail": f"{past_30} enquiries in last 30 days",
        })

    if past_12 >= 10:
        flags.append({
            "flag": "HIGH_12M_ENQUIRIES", "severity": "MEDIUM",
            "detail": f"{past_12} enquiries in last 12 months — high loan-seeking frequency",
        })

    if total >= 20:
        flags.append({
            "flag": "VERY_HIGH_TOTAL_ENQUIRIES", "severity": "MEDIUM",
            "detail": f"{total} total enquiries — extremely high lifetime loan-seeking activity",
        })

    return {
        "total":    total,
        "past_30d": past_30,
        "past_12m": past_12,
        "past_24m": past_24,
        "flags":    flags,
    }


# ============================================================
# SOURCE NOTEBOOK CELL 21
# ============================================================

def check_freshness(header_info, max_days=90):                  # change this to 90 days
    """Check if the CIBIL report is recent enough for underwriting."""
    doi = header_info.get("date_of_report")
    if not doi:
        return {
            "fresh": False, "flag": "DATE_NOT_FOUND",
            "reason": "Report date not found in document",
        }

    today = date.today()
    if doi > today:
        return {
            "fresh": False, "flag": "FUTURE_DATE",
            "reason": (
                f"Report date {doi} is in the future — "
                "possible tampered or backdated document"
            ),
        }

    age = (today - doi).days
    return {
        "fresh": age <= max_days,
        "date_of_report": str(doi),
        "age_days": age,
        "flag": None if age <= max_days else "REPORT_EXPIRED",
        "reason": (
            f"Report is {age} days old — "
            + ("OK" if age <= max_days else f"EXPIRED (max {max_days} days)")
        ),
    }


def check_identity(text, header_info):
    """
    Run CIBIL-specific identity checks.
    Returns list of flag dicts (empty = all clean).
    """
    flags = []

    # Multiple PANs in one report
    if header_info.get("multiple_pans"):
        pans = header_info.get("pans_found", [])
        flags.append({
            "flag": "MULTIPLE_PANS", "severity": "CRITICAL",
            "detail": f"Multiple PANs found in report: {pans}",
        })

    # Age checks from DOB
    dob_str = header_info.get("dob")
    if dob_str:
        try:
            dob = datetime.strptime(dob_str, "%d-%m-%Y").date()
            age_years = (date.today() - dob).days / 365.25

            if age_years < 18:
                flags.append({
                    "flag": "APPLICANT_MINOR", "severity": "CRITICAL",
                    "detail": (
                        f"DOB {dob_str} — applicant is {age_years:.1f} years old, "
                        "under 18 — cannot process loan"
                    ),
                })
            elif age_years > 70:
                flags.append({
                    "flag": "HIGH_AGE", "severity": "MEDIUM",
                    "detail": (
                        f"Applicant age {age_years:.0f} years — "
                        "may affect maximum loan tenure eligibility"
                    ),
                })
        except ValueError:
            pass

    # Gender data error (e.g. GENDER: 3)
    gender = header_info.get("gender", "")
    if "UNKNOWN" in str(gender):
        flags.append({
            "flag": "GENDER_DATA_ERROR", "severity": "LOW",
            "detail": (
                f"Non-standard gender value: {gender} — "
                "likely data entry error at originating lender, not a fraud signal"
            ),
        })

    return flags


# ============================================================
# SOURCE NOTEBOOK CELL 23
# ============================================================

def calculate_credit_vintage(account_summary, accounts):
    """How long has the person had credit? Longer = better."""
    oldest_str = account_summary.get("oldest_account")
    if not oldest_str:
        dates = []
        for acc in accounts:
            od = acc.get("opened_date")
            if od:
                try:
                    dates.append(datetime.strptime(od, "%d-%m-%Y").date())
                except ValueError:
                    pass
        if not dates:
            return {"years": 0, "risk": "UNKNOWN"}
        oldest = min(dates)
    else:
        try:
            oldest = datetime.strptime(oldest_str, "%d-%m-%Y").date()
        except ValueError:
            return {"years": 0, "risk": "UNKNOWN"}

    years = round((date.today() - oldest).days / 365.25, 1)
    risk = (
        "VERY_LOW" if years >= 5
        else "LOW"    if years >= 3
        else "MEDIUM" if years >= 1
        else "HIGH"
    )
    return {
        "years": years,
        "oldest_account": oldest_str or str(oldest),
        "risk": risk,
    }


def calculate_utilisation(accounts):
    """Overall credit utilisation across all active revolving/term accounts."""
    active = [a for a in accounts if a.get("status") == "ACTIVE"]
    total_bal  = sum(a.get("current_balance", 0) for a in active
                     if (a.get("current_balance") or 0) > 0)
    total_sanc = sum(a.get("sanctioned_amount", 0) for a in active)

    if total_sanc == 0:
        return {
            "utilization_pct": 0, "risk": "NONE",
            "total_balance": 0, "total_sanctioned": 0,
        }

    pct  = round(total_bal / total_sanc * 100, 2)
    risk = (
        "VERY_HIGH" if pct > 90
        else "HIGH"   if pct > 75
        else "MEDIUM" if pct > 50
        else "LOW"
    )
    return {
        "utilization_pct": pct, "risk": risk,
        "total_balance": total_bal, "total_sanctioned": total_sanc,
    }


def calculate_secured_unsecured(accounts):
    """Split active balance into secured vs unsecured exposure."""
    active = [a for a in accounts if a.get("status") == "ACTIVE"]
    sec_bal = unsec_bal = sec_cnt = unsec_cnt = 0

    for acc in active:
        bal = max(acc.get("current_balance", 0), 0)
        if acc.get("is_secured"):
            sec_bal += bal
            sec_cnt += 1
        else:
            unsec_bal += bal
            unsec_cnt += 1

    total = sec_bal + unsec_bal
    unsec_pct = round(unsec_bal / total * 100, 1) if total > 0 else 0

    flag = None
    if unsec_pct > 70 and total > 100000:
        flag = {
            "flag": "HIGH_UNSECURED_EXPOSURE", "severity": "MEDIUM",
            "detail": (
                f"{unsec_pct}% of active balance (₹{unsec_bal:,}) is unsecured — "
                "no collateral to recover"
            ),
        }

    return {
        "secured_balance":      sec_bal,
        "unsecured_balance":    unsec_bal,
        "secured_count":        sec_cnt,
        "unsecured_count":      unsec_cnt,
        "unsecured_pct":        unsec_pct,
        "total_active_balance": total,
        "flag":                 flag,
    }


def calculate_delinquency_recency(accounts):
    """When was the most recent bad payment? More recent = higher risk."""
    today = date.today()
    most_recent_bad = None
    most_recent_acc = None

    for acc in accounts:
        ph = acc.get("payment_history", {})
        if ph.get("worst_severity", 0) < 1:
            continue
        lr = acc.get("last_reported_date")
        if not lr:
            continue
        try:
            lr_date = datetime.strptime(lr, "%d-%m-%Y").date()
            if most_recent_bad is None or lr_date > most_recent_bad:
                most_recent_bad = lr_date
                most_recent_acc = acc["account_number"]
        except ValueError:
            pass

    if most_recent_bad is None:
        return {"has_delinquency": False, "flag": None}

    months = round((today - most_recent_bad).days / 30)

    if months <= 12:
        recency, sev = "CRITICAL", "CRITICAL"
    elif months <= 24:
        recency, sev = "HIGH", "HIGH"
    elif months <= 48:
        recency, sev = "MEDIUM", "MEDIUM"
    else:
        return {
            "has_delinquency": True,
            "months_since_last_bad": months,
            "most_recent_bad_account": most_recent_acc,
            "recency_risk": "LOW",
            "flag": None,
        }

    return {
        "has_delinquency": True,
        "months_since_last_bad": months,
        "most_recent_bad_account": most_recent_acc,
        "recency_risk": recency,
        "flag": {
            "flag": "RECENT_DELINQUENCY" if months <= 24 else "HISTORICAL_DELINQUENCY",
            "severity": sev,
            "detail": (
                f"Bad payment status {months} months ago — "
                f"account {most_recent_acc}"
            ),
        },
    }


def calculate_debt_trend(accounts):
    """
    How many new loans opened per year?
    Accelerating debt accumulation is a risk signal.
    """
    from collections import defaultdict
    yearly = defaultdict(int)

    for acc in accounts:
        od = acc.get("opened_date")
        if not od:
            continue
        try:
            year = datetime.strptime(od, "%d-%m-%Y").year
            yearly[year] += 1
        except ValueError:
            pass

    if not yearly:
        return {"trend": "UNKNOWN", "yearly_counts": {}, "flag": None}

    sorted_y = sorted(yearly.items())
    counts   = [c for _, c in sorted_y]
    flag     = None
    trend    = "STABLE"

    if len(counts) >= 3:
        if counts[-1] >= counts[-2] >= counts[-3] and counts[-1] > counts[-3]:
            trend = "ACCELERATING"
            if counts[-1] >= 3:
                flag = {
                    "flag": "DEBT_ACCELERATION", "severity": "HIGH",
                    "detail": (
                        f"Loan count increasing year over year: {dict(sorted_y)} — "
                        "rapid debt accumulation"
                    ),
                }
        elif counts[-1] < counts[-2]:
            trend = "DECELERATING"

    return {"trend": trend, "yearly_counts": dict(sorted_y), "flag": flag}


def calculate_foir_readiness(accounts):
    """
    Sum all active EMIs — needed for Fixed Obligation to Income Ratio calculation.
    Loan officer uses this + reported income to compute FOIR.
    """
    active = [a for a in accounts if a.get("status") == "ACTIVE"]
    total_emi  = sum(a.get("emi_amount", 0) for a in active)

    return {
        "total_monthly_emi": total_emi,
        "active_loan_count": len(active),
        "emi_accounts":      sum(1 for a in active if a.get("emi_amount", 0) > 0),
    }


# ============================================================
# SOURCE NOTEBOOK CELL 25
# ============================================================

def aggregate_risks(accounts, identity_flags, enquiry_result):
    """
    Combine all sources of risk flags into a single list.
    Sources: identity checks, enquiries, per-account DPD/write-off/settlement.
    """
    all_flags = list(identity_flags) + enquiry_result.get("flags", [])
    settled_list    = []
    written_off_list = []
    bad_history_list = []

    for acc in accounts:
        num    = acc["account_number"]
        atype  = acc.get("account_type", "Unknown")
        ph     = acc.get("payment_history", {})
        weight = acc.get("ownership_weight", 1.0)

        if acc.get("is_settled") and acc.get("settlement_amount", 0) > 0:
            settled_list.append(
                f"Acc {num} ({atype}) — settled ₹{acc['settlement_amount']:,}"
            )

        if acc.get("is_written_off") and acc.get("total_writeoff", 0) > 0:
            written_off_list.append(
                f"Acc {num} ({atype}) — written off ₹{acc['total_writeoff']:,}"
            )

        if ph.get("worst_severity", 0) >= 2:
            bad_history_list.append(
                f"Acc {num} ({atype}) — worst DPD: {ph['worst_dpd']} "
                f"({ph['worst_label']}), {ph['bad_months']} bad months"
                + (f" [joint/guarantor weight {weight}]" if weight < 1.0 else "")
            )

    if settled_list:
        all_flags.append({
            "flag": "SETTLED_ACCOUNTS", "severity": "HIGH",
            "detail": f"Settled accounts: {settled_list}",
            "accounts": settled_list,
        })

    if written_off_list:
        all_flags.append({
            "flag": "WRITTEN_OFF_ACCOUNTS", "severity": "CRITICAL",
            "detail": f"Written-off accounts: {written_off_list}",
            "accounts": written_off_list,
        })

    if bad_history_list:
        worst_sev = "CRITICAL" if any(
            acc.get("payment_history", {}).get("worst_severity", 0) >= 3
            for acc in accounts
        ) else "HIGH"
        all_flags.append({
            "flag": "BAD_PAYMENT_HISTORY", "severity": worst_sev,
            "detail": f"Poor payment history: {bad_history_list}",
            "accounts": bad_history_list,
        })

    return all_flags


# ============================================================
# SOURCE NOTEBOOK CELL 27
# ============================================================

SBFC_MIN_SCORE = 650


def calculate_predicted_score(accounts, bureau_score, enquiry_data=None):
    """
    Adjust the CIBIL bureau score based on actual account risk factors.
    Returns bureau score, predicted score, all adjustments, and category.
    """
    if bureau_score is None or bureau_score <= 0:
        return {
            "bureau_score":    bureau_score,
            "predicted_score": None,
            "category":        "NO_HISTORY" if bureau_score == -1 else "ERROR",
            "adjustments":     [],
            "explanation":     "No valid bureau score to adjust",
        }

    if 1 <= bureau_score <= 5:
        return {
            "bureau_score":    bureau_score,
            "predicted_score": None,
            "category":        "NEW_TO_CREDIT",
            "adjustments":     [],
            "explanation":     f"Score {bureau_score} (1-5 scale) — less than 6 months history",
        }

    score       = bureau_score
    adjustments = []
    active      = [a for a in accounts if a.get("status") == "ACTIVE"]

    # ── PENALTY 1: Write-off ──────────────────────────────────
    for acc in accounts:
        if acc.get("is_written_off") and acc.get("total_writeoff", 0) > 0:
            amt    = acc["total_writeoff"]
            weight = acc.get("ownership_weight", 1.0)
            base   = -60 if amt > 50000 else -45 if amt > 10000 else -35
            penalty = round(base * weight)
            adjustments.append({
                "factor": "WRITE_OFF", "account": acc["account_number"],
                "detail": f"Write-off ₹{amt:,} ({acc.get('account_type', '?')}) ownership weight={weight}",
                "adjustment": penalty, "severity": "CRITICAL",
            })
            score += penalty

    # ── PENALTY 2: Settlement ─────────────────────────────────
    for acc in accounts:
        if acc.get("is_settled") and acc.get("settlement_amount", 0) > 0:
            amt    = acc["settlement_amount"]
            weight = acc.get("ownership_weight", 1.0)
            base   = -35 if amt > 20000 else -25 if amt > 5000 else -15
            penalty = round(base * weight)
            adjustments.append({
                "factor": "SETTLEMENT", "account": acc["account_number"],
                "detail": f"Settled ₹{amt:,} ({acc.get('account_type', '?')})",
                "adjustment": penalty, "severity": "HIGH",
            })
            score += penalty

    # ── PENALTY 3: Bad DPD history ────────────────────────────
    for acc in accounts:
        ph     = acc.get("payment_history", {})
        worst  = ph.get("worst_severity", 0)
        bad_m  = ph.get("bad_months", 0)
        weight = acc.get("ownership_weight", 1.0)

        if worst >= 3:
            base    = max(-40 - (bad_m * 2), -70)
            penalty = round(base * weight)
            adjustments.append({
                "factor": "DBT_LOS_HISTORY", "account": acc["account_number"],
                "detail": (
                    f"Worst DPD: {ph.get('worst_dpd')} ({ph.get('worst_label')}), "
                    f"{bad_m} bad months"
                ),
                "adjustment": penalty, "severity": "CRITICAL",
            })
            score += penalty
        elif worst == 2:
            base    = max(-20 - (bad_m * 1), -40)
            penalty = round(base * weight)
            adjustments.append({
                "factor": "SUB_HISTORY", "account": acc["account_number"],
                "detail": f"Sub-standard DPD history, {bad_m} bad months",
                "adjustment": penalty, "severity": "HIGH",
            })
            score += penalty

    # ── PENALTY 4: High utilisation ───────────────────────────
    high_util = []
    for acc in active:
        sanc = acc.get("sanctioned_amount", 0)
        bal  = max(acc.get("current_balance", 0), 0)
        if sanc > 0 and bal / sanc > 0.80:
            high_util.append((acc["account_number"], round(bal / sanc * 100, 1)))

    if high_util:
        avg     = sum(u for _, u in high_util) / len(high_util)
        penalty = -20 if avg > 90 else -10
        adjustments.append({
            "factor": "HIGH_UTILISATION",
            "detail": f"High utilisation on {high_util}, avg {avg:.1f}%",
            "adjustment": penalty, "severity": "MEDIUM",
        })
        score += penalty

    # ── PENALTY 5: Enquiries ──────────────────────────────────
    if enquiry_data:
        past_12 = enquiry_data.get("past_12m", 0)
        past_30 = enquiry_data.get("past_30d", 0)

        if past_30 >= 5:
            adjustments.append({
                "factor": "HIGH_30D_ENQUIRIES",
                "detail": f"{past_30} enquiries in last 30 days",
                "adjustment": -20, "severity": "HIGH",
            })
            score -= 20
        elif past_12 >= 10:
            adjustments.append({
                "factor": "HIGH_12M_ENQUIRIES",
                "detail": f"{past_12} enquiries in last 12 months",
                "adjustment": -15, "severity": "MEDIUM",
            })
            score -= 15
        elif past_12 >= 5:
            adjustments.append({
                "factor": "MODERATE_ENQUIRIES",
                "detail": f"{past_12} enquiries in last 12 months",
                "adjustment": -8, "severity": "LOW",
            })
            score -= 8

    # ── BONUS 1: Clean active accounts ───────────────────────
    clean = [
        a for a in active
        if a.get("payment_history", {}).get("worst_severity", 0) == 0
    ]
    if clean:
        bonus = min(len(clean) * 8, 25)
        adjustments.append({
            "factor": "CLEAN_ACTIVE_ACCOUNTS",
            "detail": f"{len(clean)} active accounts with perfect payment history",
            "adjustment": +bonus, "severity": "POSITIVE",
        })
        score += bonus

    # ── BONUS 2: Zero overdue ─────────────────────────────────
    total_overdue = sum(
        a.get("current_balance", 0) for a in active
        if (a.get("current_balance") or 0) < 0
    )
    if total_overdue == 0 and active:
        adjustments.append({
            "factor": "ZERO_OVERDUE",
            "detail": "No current overdue amounts",
            "adjustment": +10, "severity": "POSITIVE",
        })
        score += 10

    # Clamp to valid CIBIL range
    score = max(300, min(900, round(score)))
    cat, risk = categorize_score(score)

    return {
        "bureau_score":     bureau_score,
        "predicted_score":  score,
        "total_adjustment": score - bureau_score,
        "category":         cat,
        "risk_level":       risk,
        "adjustments":      adjustments,
        "explanation":      (
            f"Bureau {bureau_score} → predicted {score} ({cat}) "
            f"after {len(adjustments)} adjustments"
        ),
    }


# ============================================================
# SOURCE NOTEBOOK CELL 29
# ============================================================

def make_decision(score_result, freshness, all_flags, predicted):
    """
    Apply SBFC decision waterfall.
    Returns (status_string, reason_string).
    """
    critical = [f for f in all_flags if f.get("severity") == "CRITICAL"]
    high     = [f for f in all_flags if f.get("severity") == "HIGH"]
    medium   = [f for f in all_flags if f.get("severity") == "MEDIUM"]

    # 1. Report freshness
    if not freshness.get("fresh"):
        return "REJECTED", freshness.get("reason", "Report expired or invalid date")

    # 2. CIBIL special score categories
    cat = score_result.get("category", "")
    if cat in ("NOT_IN_CIBIL", "SCORING_ERROR", "INVALID_RANGE"):
        return "MANUAL_REVIEW", score_result.get("reason", "Special score condition")

    if cat == "NEW_TO_CREDIT":
        return "MANUAL_REVIEW", score_result.get("reason", "New-to-credit applicant")

    # 3. Score not found
    if not score_result.get("found") or score_result.get("score") is None:
        return "MANUAL_REVIEW", "CIBIL score not found in document"

    # 4. Critical flags → reject
    if critical:
        names = ", ".join(f["flag"] for f in critical)
        return "REJECTED", f"Critical risk flags found: {names}"

    # 5. Score below minimum
    pred_score = (
        predicted.get("predicted_score")
        or score_result.get("score", 0)
    )
    if pred_score < SBFC_MIN_SCORE:
        return (
            "REJECTED",
            f"Predicted score {pred_score} is below SBFC minimum of {SBFC_MIN_SCORE}",
        )

    # 6. High flags → manual review
    if high:
        names = ", ".join(f["flag"] for f in high)
        return (
            "MANUAL_REVIEW",
            f"Score {pred_score} OK but high-severity flags require review: {names}",
        )

    # 7. Medium flags → approve with notes
    if medium:
        names = ", ".join(f["flag"] for f in medium)
        return (
            "APPROVED_WITH_NOTES",
            f"Score {pred_score} acceptable. Medium flags noted: {names}",
        )

    # 8. Clean
    return (
        "APPROVED",
        (
            f"Predicted score {pred_score} ({predicted.get('category', '')}) — "
            "no risk flags found"
        ),
    )


# ============================================================
# SOURCE NOTEBOOK CELL 31
# ============================================================

def generate_cibil_human_report(result):
    """
    Generate a formatted text report suitable for loan officers.
    Shows decision, score, profile, account breakdown, and all risk flags.
    """
    STATUS_EMOJI   = {
        "APPROVED":             "✅",
        "APPROVED_WITH_NOTES":  "✅⚠️",
        "MANUAL_REVIEW":        "⚠️",
        "REJECTED":             "❌",
    }
    SEVERITY_EMOJI = {
        "CRITICAL": "🔴",
        "HIGH":     "🟠",
        "MEDIUM":   "🟡",
        "LOW":      "🔵",
        "POSITIVE": "🟢",
    }
    RISK_LABEL = {
        "VERY_HIGH":  "Very High Risk",
        "HIGH":       "High Risk",
        "LOW_MEDIUM": "Low-Medium Risk",
        "MEDIUM":     "Medium Risk",
        "LOW":        "Low Risk",
        "VERY_LOW":   "Very Low Risk",
        "NONE":       "No Risk",
        "UNKNOWN":    "Unknown",
    }

    doc_check = result.get("document_type_check", {})
    status    = result.get("overall_status", "UNKNOWN")
    applicant = result.get("applicant", {})
    score     = result.get("score_analysis", {})
    predicted = score.get("predicted", {})
    credit    = result.get("credit_summary", {})
    foir      = result.get("foir_readiness", {})
    flags     = result.get("all_flags", [])
    freshness = result.get("freshness_check", {})
    enquiry   = result.get("enquiry_analysis", {})
    accounts  = result.get("account_details", [])
    accsumm   = result.get("account_summary", {})

    lines = []
    add = lines.append

    # ── Header ───────────────────────────────────────────────
    add("=" * 65)
    add("  CIBIL CREDIT VERIFICATION REPORT — SBFC Finance")
    add("=" * 65)
    add("")
    add(f"  Document Type : {doc_check.get('detected','Unknown')}")
    add(f"  File Name     : {result.get('file','—')}")
    add(f"  Verified At   : {result.get('verified_at','—')}")
    add(f"  DECISION  :  {STATUS_EMOJI.get(status, '❓')}  {status}")
    add(f"  REASON    :  {result.get('decision_reason', '')}")
    add("")

    # ── Applicant ────────────────────────────────────────────
    add("-" * 65)
    add("  APPLICANT")
    add("-" * 65)
    add(f"  Name           : {applicant.get('name', '—')}")
    add(f"  PAN            : {applicant.get('pan', '—')}")
    add(f"  Date of Birth  : {applicant.get('dob', '—')}  |  "
        f"Gender: {applicant.get('gender', '—')}")
    if applicant.get("income_monthly"):
        add(f"  Reported Income: ₹{applicant['income_monthly']:,} / month "
            f"({applicant.get('income_type', '')})")
    add(f"  CIBIL Ref      : {applicant.get('control_number', '—')}")
    add(f"  Report Date    : {freshness.get('date_of_report', '—')}  "
        f"({freshness.get('age_days', '?')} days old)")
    if applicant.get("multiple_pans"):
        add(f"  ⚠️  MULTIPLE PANs FOUND: {applicant.get('all_pans')}")
    add("")

    # ── Score ────────────────────────────────────────────────
    add("-" * 65)
    add("  CREDIT SCORE")
    add("-" * 65)
    bureau_score = score.get("score")
    pred_score   = predicted.get("predicted_score")
    factors      = score.get("scoring_factors", [])

    if bureau_score and bureau_score > 5:
        add(f"  Bureau Score (CIBIL) : {bureau_score}  "
            f"({score.get('category', '—')}  |  "
            f"{RISK_LABEL.get(score.get('risk_level', ''), '—')})")

        if pred_score:
            adj = predicted.get("total_adjustment", 0)
            add(f"  Predicted Score      : {pred_score}  "
                f"({predicted.get('category', '—')}  |  "
                f"{RISK_LABEL.get(predicted.get('risk_level', ''), '—')})  "
                f"[{adj:+d} pts]")
            # add(f"  SBFC Minimum         : {SBFC_MIN_SCORE}  "
            #     + ("✅ PASS" if pred_score >= SBFC_MIN_SCORE else "❌ FAIL"))

        if factors:
            add("")
            add("  CIBIL scoring factors (why the score is what it is):")
            for f in factors[:4]:
                add(f"    • {f.strip()}")

        # if predicted.get("adjustments"):
        #     add("")
        #     add("  Lender-side score adjustments:")
        #     for adj in predicted["adjustments"]:
        #         e        = SEVERITY_EMOJI.get(adj["severity"], "⚪")
        #         acc_info = f" [Acc {adj['account']}]" if adj.get("account") else ""
        #         label    = adj["factor"].replace("_", " ").title()
        #         add(f"    {e}  {adj['adjustment']:+d} pts  {label}{acc_info}")
        #         add(f"           ↳ {adj.get('detail', '')}")

    elif bureau_score == -1:
        add("  Score : -1 — Consumer NOT in CIBIL database")
        add("          No credit accounts found in CIBIL.")
        add("          Could be genuinely new to formal credit.")
        add("          ⚠️  Manual evaluation required.")
    elif bureau_score is not None and 1 <= bureau_score <= 5:
        add(f"  Score : {bureau_score} (1-5 scale) — Less than 6 months credit history")
        add("          Thin-file case — manual evaluation required.")
    else:
        add("  Score : Not sfound in document")
    add("")

    # ── Credit Profile ────────────────────────────────────────
    add("-" * 65)
    add("  CREDIT PROFILE")
    add("-" * 65)
    vintage = credit.get("credit_vintage", {})
    util    = credit.get("credit_utilization", {})
    exp     = credit.get("secured_unsecured_exposure", {})
    delinq  = credit.get("delinquency_recency", {})
    dtrent  = credit.get("debt_trend", {})

    add(f"  Total Accounts        : {accsumm.get('total_accounts', '—')}  "
        f"|  Zero-balance: {accsumm.get('zero_balance_accounts', '—')}")
    add(f"  Credit History Age    : {vintage.get('years', '—')} yrs  "
        f"(oldest: {vintage.get('oldest_account', '—')})  "
        f"— {RISK_LABEL.get(vintage.get('risk', ''), '—')}")
    add(f"  Portfolio Utilisation : {util.get('utilization_pct', '—')}%  "
        f"— {RISK_LABEL.get(util.get('risk', ''), '—')}")
    add(f"  Active EMI Burden     : ₹{foir.get('total_monthly_emi', 0):,} / month  "
        f"({foir.get('active_loan_count', 0)} active loans, "
        f"{foir.get('emi_accounts', 0)} with EMI)")
    add(f"  Secured Exposure      : ₹{exp.get('secured_balance', 0):,}  "
        f"({exp.get('secured_count', 0)} accounts 🔒)")
    add(f"  Unsecured Exposure    : ₹{exp.get('unsecured_balance', 0):,}  "
        f"({exp.get('unsecured_pct', 0)}%)")
    add(f"  Debt Trend            : {dtrent.get('trend', '—')}  "
        f"— yearly: {dtrent.get('yearly_counts', {})}")

    if delinq.get("has_delinquency"):
        add(f"  Last Bad Payment      : {delinq.get('months_since_last_bad', '?')} months ago  "
            f"(Acc {delinq.get('most_recent_bad_account', '?')})  "
            f"— {RISK_LABEL.get(delinq.get('recency_risk', ''), '')}")
    else:
        add("  Last Bad Payment      : None found ✅")

    add(f"  Enquiries             : {enquiry.get('total', '?')} total  |  "
        f"{enquiry.get('past_30d', '?')} in 30 days  |  "
        f"{enquiry.get('past_12m', '?')} in 12 months  |  "
        f"{enquiry.get('past_24m', '?')} in 24 months")
    add("")

    # ── Account Breakdown ─────────────────────────────────────
    add("-" * 65)
    add("  ACCOUNT BREAKDOWN")
    add("-" * 65)
    active_accs  = [a for a in accounts if a.get("status") == "ACTIVE"]
    closed_accs  = [a for a in accounts if a.get("status") == "CLOSED"]
    problem_accs = [a for a in accounts if a.get("has_issues")]

    add(f"  Total: {len(accounts)}  |  "
        f"Active: {len(active_accs)}  |  "
        f"Closed: {len(closed_accs)}  |  "
        f"Problem: {len(problem_accs)}")
    add("")

    if active_accs:
        add("  Active Accounts:")
        for acc in active_accs:
            sanc  = acc.get("sanctioned_amount", 0)
            bal   = max(acc.get("current_balance", 0), 0)
            util_str = f"  Util: {round(bal / sanc * 100)}%" if sanc > 0 else ""
            emi_str  = f"  EMI: ₹{acc.get('emi_amount', 0):,}" if acc.get("emi_amount") else ""
            sec_icon = "🔒" if acc.get("is_secured") else ""
            own      = acc.get("ownership", "?")
            add(f"    ✅{sec_icon}  Acc {acc['account_number']:>2}  "
                f"{acc.get('account_type', '?'):<30}  "
                f"({own})  Bal: ₹{bal:>12,}{util_str}{emi_str}")

    if problem_accs:
        add("")
        add("  Problem Accounts:")
        for acc in problem_accs:
            ph = acc.get("payment_history", {})
            add(f"    🔴  Acc {acc['account_number']:>2}  "
                f"{acc.get('account_type', '?'):<30}  "
                f"({acc.get('ownership', '?')})  "
                f"Opened: {acc.get('opened_date', '?')}  "
                f"Closed: {acc.get('closed_date', 'Open')}")
            if acc.get("is_written_off"):
                add(f"          → Written-off ₹{acc.get('total_writeoff', 0):,}  "
                    f"Status: {acc.get('writeoff_status', '')}")
            if acc.get("is_settled"):
                add(f"          → Settled ₹{acc.get('settlement_amount', 0):,}")
            if ph.get("worst_severity", 0) >= 2:
                add(f"          → Worst DPD: {ph['worst_dpd']} ({ph['worst_label']})  "
                    f"Bad months: {ph['bad_months']}")
    add("")

    # ── Risk Flags ───────────────────────────────────────────
    add("-" * 65)
    add("  RISK FLAGS")
    add("-" * 65)

    if not flags:
        add("  ✅  No risk flags — document and credit profile are clean")
    else:
        shown = 0
        for sev, label in [
            ("CRITICAL", "CRITICAL"),
            ("HIGH",     "HIGH"),
            ("MEDIUM",   "MEDIUM"),
            ("LOW",      "NOTE"),
        ]:
            for flag in [f for f in flags if f.get("severity") == sev]:
                e    = SEVERITY_EMOJI.get(sev, "⚪")
                name = flag.get("flag", "").replace("_", " ").title()
                add(f"  {e}  {label}: {name}")
                add(f"      {flag.get('detail', '')}")
                shown += 1
        if shown == 0:
            add("  🟢  Only positive / informational flags")

    add("")
    add("=" * 65)
    add("  END OF REPORT")
    add("=" * 65)

    return "\n".join(lines)


# ============================================================
# SOURCE NOTEBOOK CELL 33
# ============================================================

def verify_cibil_document(file_bytes, applicant_pan=None, filename="cibil_report.pdf"):
    """
    Full CIBIL verification pipeline.

    Validation/classification is performed before expensive credit analysis.
    OCR is intentionally disabled. Native PDFs continue through the existing
    CIBIL business engine; scanned/image-only PDFs are sent to REVIEW because
    reliable document classification requires text.
    """

    started = __import__("time").perf_counter()

    result = {
        "file": os.path.basename(filename or "cibil_report.pdf"),
        "bureau": "CIBIL",
        "verified_at": datetime.now().isoformat(),
        "overall_status": None,
        "decision_reason": None,
        "applicant": {},
        "score_analysis": {},
        "account_summary": {},
        "account_details": [],
        "credit_summary": {},
        "foir_readiness": {},
        "identity_check": {},
        "enquiry_analysis": {},
        "freshness_check": {},
        "all_flags": [],
        "risk_summary": {},
        "human_report": "",
    }

    # ------------------------------------------------------------
    # 1. PDF security / structure
    # ------------------------------------------------------------
    fc = check_file(file_bytes)
    result["file_check"] = fc

    if not fc["passed"]:
        result["overall_status"] = "REJECTED"
        result["decision_reason"] = fc["reason"]
        result["processing_time_ms"] = round((__import__("time").perf_counter() - started) * 1000, 2)
        return result

    # ------------------------------------------------------------
    # 2. Native text extraction — OCR intentionally disabled
    # ------------------------------------------------------------
    full_text = extract_full_text(file_bytes)
    result["native_text"] = bool(full_text.strip())

    # ------------------------------------------------------------
    # 3. Document classification BEFORE CIBIL processing
    # ------------------------------------------------------------
    doc_type = detect_document_type(full_text)
    result["document_type_check"] = doc_type

    # Scanned / image-only: no OCR. Do not falsely call it CIBIL.
    if not full_text.strip():
        result["overall_status"] = "MANUAL_REVIEW"
        result["decision_reason"] = (
            "No native text layer found. OCR is disabled, so the document "
            "cannot be reliably classified as a CIBIL report."
        )
        result["processing_time_ms"] = round((__import__("time").perf_counter() - started) * 1000, 2)
        return result

    # Known wrong document -> reject immediately.
    if doc_type["detected"] != "CIBIL":
        result["overall_status"] = "REJECTED"
        result["decision_reason"] = (
            f"Wrong document uploaded. Detected {doc_type['detected']} "
            "instead of a CIBIL report."
        )
        result["processing_time_ms"] = round((__import__("time").perf_counter() - started) * 1000, 2)
        return result

    # ------------------------------------------------------------
    # 4. Bureau validation
    # ------------------------------------------------------------
    bureau = detect_bureau(full_text)
    result["bureau_check"] = bureau

    if not bureau["is_cibil"]:
        result["overall_status"] = "REJECTED"
        result["decision_reason"] = bureau.get(
            "reason",
            "Not a CIBIL document",
        )
        result["processing_time_ms"] = round((__import__("time").perf_counter() - started) * 1000, 2)
        return result

    # ------------------------------------------------------------
    # 5. Existing CIBIL business engine
    # ------------------------------------------------------------
    header = extract_header_info(full_text)
    result["applicant"] = {
        "name": header.get("name"),
        "pan": header.get("pan"),
        "all_pans": header.get("pans_found", []),
        "dob": header.get("dob"),
        "gender": header.get("gender"),
        "multiple_pans": header.get("multiple_pans", False),
        "income_monthly": header.get("income_monthly"),
        "income_type": header.get("income_type"),
        "control_number": header.get("control_number"),
        "member_ref": header.get("member_ref"),
    }

    # PAN cross-check is active when the caller supplies PAN.
    if applicant_pan and header.get("pan"):
        supplied_pan = re.sub(r"\s+", "", applicant_pan.upper())
        report_pan = re.sub(r"\s+", "", header["pan"].upper())
        if supplied_pan != report_pan:
            result["overall_status"] = "REJECTED"
            result["decision_reason"] = (
                f"PAN mismatch — report: {report_pan}, "
                f"application: {supplied_pan}"
            )
            result["processing_time_ms"] = round((__import__("time").perf_counter() - started) * 1000, 2)
            return result

    score_raw = extract_score(full_text)
    score_raw["scoring_factors"] = extract_scoring_factors(full_text[:2500])

    if (
        score_raw.get("found")
        and score_raw.get("score") is not None
        and score_raw["score"] > 5
        and "category" not in score_raw
    ):
        cat, risk = categorize_score(score_raw["score"])
        score_raw["category"] = cat
        score_raw["risk_level"] = risk

    result["score_analysis"] = score_raw
    result["account_summary"] = extract_account_summary(full_text)

    accounts = extract_all_accounts(full_text)
    result["account_details"] = accounts

    identity_flags = check_identity(full_text, header)
    result["identity_check"] = {
        "flags": identity_flags,
        "has_issues": len(identity_flags) > 0,
    }

    enq = analyse_enquiries(full_text)
    result["enquiry_analysis"] = enq

    freshness = check_freshness(header)
    result["freshness_check"] = freshness

    vintage = calculate_credit_vintage(result["account_summary"], accounts)
    util = calculate_utilisation(accounts)
    exposure = calculate_secured_unsecured(accounts)
    delinq = calculate_delinquency_recency(accounts)
    debt_trend = calculate_debt_trend(accounts)

    result["credit_summary"] = {
        "credit_vintage": vintage,
        "credit_utilization": util,
        "secured_unsecured_exposure": exposure,
        "delinquency_recency": delinq,
        "debt_trend": debt_trend,
    }

    result["foir_readiness"] = calculate_foir_readiness(accounts)

    all_flags = aggregate_risks(accounts, identity_flags, enq)
    for extra in [
        exposure.get("flag"),
        delinq.get("flag"),
        debt_trend.get("flag"),
    ]:
        if extra:
            all_flags.append(extra)
    result["all_flags"] = all_flags

    predicted = calculate_predicted_score(
        accounts,
        score_raw.get("score"),
        enq,
    )
    result["score_analysis"]["predicted"] = predicted

    status, reason = make_decision(
        score_raw,
        freshness,
        all_flags,
        predicted,
    )
    result["overall_status"] = status
    result["decision_reason"] = reason

    result["risk_summary"] = {
        "total_flags": len(all_flags),
        "critical_flags": [
            f["flag"] for f in all_flags
            if f.get("severity") == "CRITICAL"
        ],
        "high_flags": [
            f["flag"] for f in all_flags
            if f.get("severity") == "HIGH"
        ],
        "medium_flags": [
            f["flag"] for f in all_flags
            if f.get("severity") == "MEDIUM"
        ],
        "accounts_with_issues": [
            a["account_number"] for a in accounts
            if a.get("has_issues")
        ],
        "total_accounts_analysed": len(accounts),
        "settled_count": sum(
            1 for a in accounts if a.get("is_settled")
        ),
        "written_off_count": sum(
            1 for a in accounts if a.get("is_written_off")
        ),
        "predicted_score": predicted.get("predicted_score"),
    }

    result["human_report"] = generate_cibil_human_report(result)
    result["processing_time_ms"] = round((__import__("time").perf_counter() - started) * 1000, 2)
    return result
