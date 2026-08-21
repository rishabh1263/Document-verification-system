import pdfplumber
import re
import os
import json
from datetime import datetime, date

def detect_document_type(text):
    """
    Detect what type of financial document this is before processing.
    Returns: "CIBIL" | "CRIF" | "BANK_STATEMENT" | "UNKNOWN"
    
    Called first in every verifier so we can reject wrong document type
    early — e.g. someone uploads a CRIF report to the CIBIL endpoint.
    """
    text_upper = text.upper()

    # CIBIL signatures
    cibil_hits = sum(1 for kw in [
        "TRANSUNION CIBIL", "CIBILTUSC3", "CONSUMER CIR",
        "CIBIL TRANSUNION SCORE"
    ] if kw in text_upper)

    # CRIF signatures
    crif_hits = sum(1 for kw in [
        "CRIF HIGH MARK", "CRIF HM SCORE", "PERFORM CONSUMER",
        "CREDIT INFORMATION REPORT PROV2", "CHM REF #"
    ] if kw in text_upper)

    # Bank statement signatures
    bank_hits = sum(1 for kw in [
        "ACCOUNT STATEMENT", "BANK STATEMENT", "STATEMENT OF ACCOUNT",
        "OPENING BALANCE", "CLOSING BALANCE",
        "DEBIT", "CREDIT", "WITHDRAWAL", "DEPOSIT"
    ] if kw in text_upper)

    scores = {
        "CIBIL":          cibil_hits,
        "CRIF":           crif_hits,
        "BANK_STATEMENT": bank_hits,
    }

    best      = max(scores, key=scores.get)
    best_hits = scores[best]

    # Need at least 2 matches to be confident
    if best_hits < 2:
        return {
            "detected":   "UNKNOWN",
            "confidence": "NONE",
            "scores":     scores,
            "reason":     "Document does not match CIBIL, CRIF, or bank statement patterns"
        }

    return {
        "detected":   best,
        "confidence": "HIGH" if best_hits >= 3 else "MEDIUM",
        "scores":     scores,
        "reason":     f"Detected as {best} ({best_hits} keyword matches)"
    }


def check_file(filepath):
    result = {"passed": False, "reason": ""}
    try:
        with open(filepath, 'rb') as f:
            if not f.read(5).startswith(b'%PDF'):
                result["reason"] = "Not a valid PDF (wrong header)"
                return result
        with pdfplumber.open(filepath) as pdf:
            pages = len(pdf.pages)
            if pages == 0:
                result["reason"] = "PDF has 0 pages"
                return result
            total_chars = sum(len(p.extract_text() or "") for p in pdf.pages)
            if total_chars < 100:
                result["reason"] = "No extractable text — upload digital PDF from CRIF website, not a photo/scan"
                return result
            result.update({"passed": True, "pages": pages, "chars": total_chars})
    except Exception as e:
        result["reason"] = f"Cannot open PDF: {e}"
    return result

def extract_full_text(filepath):
    full = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            full += (page.extract_text() or "") + "\n"
    return full

def extract_header_info(text):
    info = {}

    m = re.search(r'CHM Ref #:\s*(\S+)', text)
    if m: info["chm_ref"] = m.group(1)

    m = re.search(r'Application ID:\s*(\S+)', text)
    if m: info["application_id"] = m.group(1)

    m = re.search(r'Prepared For:\s*(.+?)(?:\n|Application)', text)
    if m: info["prepared_for"] = m.group(1).strip()

    # Date of Issue = report generation date
    m = re.search(r'Date of Issue:\s*(\d{2}-\d{2}-\d{4})', text)
    if m:
        info["date_of_issue_raw"] = m.group(1)
        try:
            info["date_of_issue"] = datetime.strptime(m.group(1), "%d-%m-%Y").date()
        except: pass

    m = re.search(r'For\s+([A-Z][A-Z\s]+?)\n', text)
    if m: info["applicant_name"] = m.group(1).strip()

    # All PANs in document — multiple = identity risk
    pans = re.findall(r'([A-Z]{5}[0-9]{4}[A-Z])\s*\[PAN\]', text.upper())
    unique_pans = list(set(pans))
    info["pans_found"]    = unique_pans
    info["pan"]           = unique_pans[0] if len(unique_pans) == 1 else None
    info["multiple_pans"] = len(unique_pans) > 1

    m = re.search(r'DOB/Age:\s*(\d{2}-\d{2}-\d{4})', text)
    if m: info["dob"] = m.group(1)

    m = re.search(r'Gender:\s*(MALE|FEMALE|OTHER)', text.upper())
    if m: info["gender"] = m.group(1)

    return info

def extract_score(text):
    # Primary: PERFORM CONSUMER row → number after 300-900
    m = re.search(r'PERFORM\s+CONSUMER[\s\S]{0,50}?300-900\s+(\d{3})', text.upper())
    if m:
        score = int(m.group(1))
        if 300 <= score <= 900:
            return {"found": True, "score": score, "source": "PERFORM_CONSUMER_ROW"}

    # Secondary: any 300-900 followed by 3-digit number
    m = re.search(r'300-900\s+(\d{3})', text.upper())
    if m:
        score = int(m.group(1))
        if 300 <= score <= 900:
            return {"found": True, "score": score, "source": "RANGE_PATTERN"}

    # PERFORM row exists but score is blank → genuine no history
    if re.search(r'PERFORM\s+CONSUMER', text.upper()):
        return {"found": False, "score": None, "category": "NO_HISTORY",
                "reason": "PERFORM CONSUMER row found but no score — applicant has no credit history (NH)"}

    return {"found": False, "score": None, "reason": "Score section not found"}


def categorize_score(score):
    if score >= 800: return "EXCELLENT",      "VERY_LOW"
    if score >= 750: return "GOOD",           "LOW"
    if score >= 650: return "AVERAGE",        "MEDIUM"
    if score >= 550: return "BELOW_AVERAGE",  "HIGH"
    return "POOR", "VERY_HIGH" 

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

STATUS_SEVERITY = {
    "STD": 0, "XXX": 0,  # Normal / no data
    "SMA": 1,             # Special mention — watch
    "SUB": 2,             # Sub-standard — bad
    "DBT": 3,             # Doubtful — very bad
    "LOS": 4,             # Loss — critical
}

CRIF_TO_DPD = {"STD": 0, "SMA": 15, "SUB": 60, "DBT": 150, "LOS": 270, "XXX": 0}

def parse_payment_history(text_block):
    statuses = re.findall(r'(?:\d{3}|XXX)/(\w{3})', text_block)
    if not statuses:
        return {"worst_status": "UNKNOWN", "worst_severity": -1, "bad_months_count": 0}

    worst_severity = -1
    worst_status   = "STD"
    bad_count      = 0

    for s in statuses:
        sev = STATUS_SEVERITY.get(s, 0)
        if sev > worst_severity:
            worst_severity = sev
            worst_status   = s
        if sev >= 1:
            bad_count += 1

    return {
        "worst_status":    worst_status,
        "worst_severity":  worst_severity,
        "bad_months_count": bad_count,
        "all_statuses":    list(set(statuses)),
        "rbi_classification": get_rbi_classification(CRIF_TO_DPD.get(worst_status, 0))
    }

def parse_single_account(acc_num, block):
    acc = {"account_number": acc_num}

    m = re.search(r'Account Type:\s*(.+?)(?:Credit Grantor|$)', block, re.IGNORECASE)
    if m: acc["account_type"] = m.group(1).strip()

    m = re.search(r'Lender Type:\s*(\w+)', block)
    if m: acc["lender_type"] = m.group(1)

    m = re.search(r'Ownership:\s*(\w+)', block)
    if m: acc["ownership"] = m.group(1)

    # if re.search(r'\bActive\b', block):    acc["status"] = "ACTIVE"
    # elif re.search(r'\bClosed\b', block):  acc["status"] = "CLOSED"
    # else: acc["status"] = "UNKNOWN"
    
    closed_date_match = re.search(r'Closed Date:\s*(\d{2}-\d{2}-\d{4})', block)
    if closed_date_match:
        acc["status"]      = "CLOSED"
        acc["closed_date"] = closed_date_match.group(1)
    else:
        # No actual date in Closed Date field → account is still open
        acc["status"] = "ACTIVE"

    m = re.search(r'Disbd Amt/High Credit:\s*([\d,]+)', block)
    if m: acc["disbursed_amount"] = int(m.group(1).replace(',',''))

    m = re.search(r'Credit Limit:\s*([\d,]+)', block)
    if m:
        val = m.group(1).replace(',', '').strip()
        acc["credit_limit"] = int(val) if val else 0

    # Last reported date — used for delinquency recency calculation
    m = re.search(r'As on:\s*(\d{2}-\d{2}-\d{4})', block)
    if m: acc["last_reported_date"] = m.group(1)

    m = re.search(r'Current Balance:\s*([\d,]+)', block)
    if m: acc["current_balance"] = int(m.group(1).replace(',',''))

    m = re.search(r'Overdue Amt:\s*([\d,]*)', block)
    if m:
        val = m.group(1).replace(',','').strip()
        acc["overdue_amount"] = int(val) if val else 0

    # Settlement — KEY FLAG: paid less than full amount
    m = re.search(r'Settlement Amt:\s*([\d,]+)', block)
    if m:
        val = int(m.group(1).replace(',',''))
        if val > 0:
            acc["settlement_amount"] = val
            acc["is_settled"] = True
        else:
            acc["is_settled"] = False
    else:
        acc["is_settled"] = False

    # Write-off — KEY FLAG: lender gave up recovering
    m = re.search(r'Total Writeoff Amt:\s*([\d,]+)', block)
    if m:
        val = int(m.group(1).replace(',',''))
        acc["total_writeoff"]  = val
        acc["is_written_off"]  = val > 0
    else:
        acc["is_written_off"] = False

    # Account remarks
    m = re.search(r'Account Remarks:\s*(.+?)(?:Income|$)', block, re.IGNORECASE | re.DOTALL)
    if m:
        remark = m.group(1).strip()
        acc["account_remarks"] = remark
        remark_up = remark.upper()
        if "SUIT FILED" in remark_up and "NO SUIT" not in remark_up:
            acc["suit_filed"] = True

    m = re.search(r'Interest Rate:\s*([\d.]+)\s*%', block)
    if m: acc["interest_rate"] = float(m.group(1))

    m = re.search(r'Disbursed Date:\s*(\d{2}-\d{2}-\d{4})', block)
    if m: acc["disbursed_date"] = m.group(1)

    # foir analysis
    emi_match = re.search(
        r"InstlAmt/Freq:\s*([\d,]+)",
        block,
        re.I
    )
    emi_amount = (
        int(emi_match.group(1).replace(",", ""))
        if emi_match
        else 0
    )
    acc["emi_amount"] = emi_amount

    ph = parse_payment_history(block)
    acc["payment_history"] = ph

    # Account-level issue flag
    acc["has_issues"] = (
        acc.get("is_settled", False) or
        acc.get("is_written_off", False) or
        acc.get("suit_filed", False) or
        ph["worst_severity"] >= 2
    )
    return acc


def extract_all_accounts(full_text):
    parts = re.split(r'Account Information\s*\n\s*(\d+)\s*\n?', full_text)
    accounts = []
    i = 1
    while i < len(parts) - 1:
        acc = parse_single_account(parts[i], parts[i+1])
        accounts.append(acc)
        i += 2
    return accounts


def calculate_credit_vintage(accounts):
    """
    Measures how long the applicant has been using credit.

    Why it matters:
    - Longer credit history generally indicates lower risk.
    - Borrowers with 8-10 years of repayment history are more predictable.
    - Very new borrowers (<2 years) have limited repayment track record.

    CRIF Source:
    - Uses earliest 'Disbursed Date' found across all accounts.

    Example:
    Account 1 opened: 2018
    Account 2 opened: 2022

    Vintage = 2018 → Today = ~8 years

    Returns:
    {
        "years": 8.1,
        "oldest_account": "15-11-2018",
        "risk": "LOW"
    }
    """
    valid_dates = []
    for acc in accounts:
        dt = acc.get("disbursed_date")
        if not dt:
            continue
        try:
            valid_dates.append(
                datetime.strptime(dt, "%d-%m-%Y").date()
            )
        except:
            pass

    if not valid_dates:
        return {
            "years": 0,
            "oldest_account": None,
            "risk": "UNKNOWN"
        }

    oldest = min(valid_dates)

    years = round(
        (date.today() - oldest).days / 365,
        1
    )

    if years >= 10:
        risk = "VERY_LOW"
    elif years >= 5:
        risk = "LOW"
    elif years >= 2:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "years": years,
        "oldest_account": oldest.strftime("%d-%m-%Y"),
        "risk": risk
    }


def calculate_secured_unsecured_exposure(accounts):
    """
    Splits active balance into secured vs unsecured.

    Why it matters:
    - Secured loans (property, vehicle) have collateral backing —
      lender can recover money if person defaults
    - Unsecured loans (personal, business, consumer) have no backing —
      total loss if person defaults
    - If 70%+ of active balance is unsecured → high exposure risk for lenders

    Secured account types:
    Property Loan, Home Loan, Vehicle Loan, Gold Loan, Loan Against Property

    Unsecured account types:
    Personal Loan, Business Loan, Consumer Loan, MFI loans
    """
    SECURED_TYPES = [
        "PROPERTY LOAN", "HOME LOAN", "MORTGAGE",
        "VEHICLE", "CAR LOAN", "TWO-WHEELER",
        "GOLD LOAN", "LOAN AGAINST"
    ]

    secured_bal = unsecured_bal = 0
    secured_count = unsecured_count = 0

    active = [a for a in accounts if a.get("status") == "ACTIVE"]

    for acc in active:
        atype  = acc.get("account_type", "").upper()
        bal    = acc.get("current_balance", 0)
        is_sec = any(s in atype for s in SECURED_TYPES)

        if is_sec:
            secured_bal   += bal
            secured_count += 1
        else:
            unsecured_bal   += bal
            unsecured_count += 1

    total         = secured_bal + unsecured_bal
    unsecured_pct = round(unsecured_bal / total * 100, 1) if total > 0 else 0

    flag = None
    if unsecured_pct > 70 and total > 100000:
        flag = {
            "flag":     "HIGH_UNSECURED_EXPOSURE",
            "severity": "MEDIUM",
            "detail":   f"{unsecured_pct}% of active balance (₹{unsecured_bal:,}) "
                        f"is unsecured — no collateral backing"
        }

    return {
        "secured_balance":      secured_bal,
        "unsecured_balance":    unsecured_bal,
        "secured_count":        secured_count,
        "unsecured_count":      unsecured_count,
        "unsecured_pct":        unsecured_pct,
        "total_active_balance": total,
        "flag":                 flag
    }

def calculate_delinquency_recency(accounts):
    """
    Answers: when was the LAST time any account had a bad payment status?

    Why it matters:
    - DBT 6 months ago = person is currently in trouble → CRITICAL
    - DBT 2 years ago  = recovering, still risky → HIGH
    - DBT 4 years ago  = old history, mostly forgiven → MEDIUM
    - DBT 6+ years ago = too old to matter much → no flag

    CRIF's 'As on' date on each account = last time data was reported.
    We use this as a proxy for when the bad status last occurred.

    Returns:
    {
        "has_delinquency": True,
        "months_since_last_bad": 28,
        "most_recent_bad_account": "6",
        "recency_risk": "MEDIUM",
        "flag": {...} or None
    }
    """
    STATUS_SEV = {"STD":0,"XXX":0,"SMA":1,"SUB":2,"DBT":3,"LOS":4}
    today = date.today()

    most_recent_bad_date = None
    most_recent_bad_acc  = None

    for acc in accounts:
        ph = acc.get("payment_history", {})
        if ph.get("worst_severity", 0) < 1:
            continue   # this account has no bad history, skip

        # 'As on' date from report = last time this account was reported
        lr = acc.get("last_reported_date")
        if not lr:
            continue
        try:
            lr_date = datetime.strptime(lr, "%d-%m-%Y").date()
            if most_recent_bad_date is None or lr_date > most_recent_bad_date:
                most_recent_bad_date = lr_date
                most_recent_bad_acc  = acc["account_number"]
        except:
            pass

    if most_recent_bad_date is None:
        return {
            "has_delinquency":        False,
            "months_since_last_bad":  None,
            "recency_risk":           "NONE",
            "flag":                   None
        }

    months_ago = round((today - most_recent_bad_date).days / 30)

    if months_ago <= 12:
        recency_risk = "CRITICAL"
        flag = {
            "flag":     "RECENT_DELINQUENCY",
            "severity": "CRITICAL",
            "detail":   f"Bad payment status within last {months_ago} months "
                        f"(account {most_recent_bad_acc}). Active delinquency risk."
        }
    elif months_ago <= 24:
        recency_risk = "HIGH"
        flag = {
            "flag":     "RECENT_DELINQUENCY",
            "severity": "HIGH",
            "detail":   f"Bad payment status {months_ago} months ago "
                        f"(account {most_recent_bad_acc})."
        }
    elif months_ago <= 48:
        recency_risk = "MEDIUM"
        flag = {
            "flag":     "HISTORICAL_DELINQUENCY",
            "severity": "MEDIUM",
            "detail":   f"Bad payment status {months_ago} months ago "
                        f"(account {most_recent_bad_acc}). Older history."
        }
    else:
        recency_risk = "LOW"
        flag = None   # too old to raise a flag

    return {
        "has_delinquency":         True,
        "months_since_last_bad":   months_ago,
        "most_recent_bad_account": most_recent_bad_acc,
        "recency_risk":            recency_risk,
        "flag":                    flag
    }


def calculate_credit_utilization(accounts):
    """
    Portfolio Credit Utilization Analysis

    Measures how much of available credit the applicant is currently using.

    Why it matters:
    - High utilization indicates financial stress.
    - Borrowers consistently using >75% of available credit
      are statistically more likely to default.
    - Lower utilization indicates healthier credit behaviour.

    Formula:
        Total Current Balance
        --------------------- X 100
        Total Exposure

    Exposure =
    Credit Limit (preferred)
    OR
    Disbursed Amount / High Credit

    Example:

    Account A:
        Limit = 100,000
        Balance = 50,000

    Account B:
        Limit = 200,000
        Balance = 100,000

    Utilization =
        (150,000 / 300,000) * 100
        = 50%

    Returns:
    {
        "utilization_pct": 50.0,
        "risk": "MEDIUM",
        "total_balance": 150000,
        "total_exposure": 300000
    }
    """

    total_exposure = 0
    total_balance = 0

    for acc in accounts:

        # Only active accounts contribute to current utilization
        if acc.get("status") != "ACTIVE":
            continue

        limit = (
            acc.get("credit_limit", 0)
            or acc.get("disbursed_amount", 0)
        )

        balance = acc.get("current_balance", 0)

        if limit <= 0:
            continue

        total_exposure += limit
        total_balance += balance

    if total_exposure == 0:
        return {
            "utilization_pct": 0,
            "risk": "LOW"
        }

    util = round(
        total_balance * 100 / total_exposure,
        2
    )

    if util >= 90:
        risk = "VERY_HIGH"
    elif util >= 75:
        risk = "HIGH"
    elif util >= 50:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "utilization_pct": util,
        "risk": risk,
        "total_balance": total_balance,
        "total_exposure": total_exposure
    }


"""
    FOIR = Fixed Obligation to Income Ratio
    Formula:FOIR = (Total Monthly Obligations / Monthly Income) × 100
    It tells: Can he afford to repay?
"""
def calculate_emi_profile(accounts):
    """
    CRIF-side EMI analysis.
    Used later for FOIR calculation.
    """

    active_accounts = [
        a for a in accounts
        if a.get("status") == "ACTIVE"
    ]

    total_emi = 0
    emi_accounts = 0

    for acc in active_accounts:

        emi = acc.get("emi_amount", 0)

        if emi and emi > 0:
            total_emi += emi
            emi_accounts += 1

    return {
        "total_monthly_emi": round(total_emi),
        "active_loan_count": len(active_accounts),
        "emi_accounts": emi_accounts,
        "high_emi_burden": total_emi > 50000
    }

def check_identity_consistency(text):
    flags = []

    pans_in_ids = re.findall(r'([A-Z]{5}[0-9]{4}[A-Z])\s*\[PAN\]', text.upper())
    unique_pans = list(set(pans_in_ids))
    if len(unique_pans) > 1:
        flags.append({
            "flag": "MULTIPLE_PANS",
            "severity": "CRITICAL",
            "detail": f"Multiple PANs in report: {unique_pans}. Possible identity mixing or fraud.",
            "values": unique_pans
        })

    # dob_section = re.search(r'DOB Variations(.+?)(?:Phone Variations|Address Variations|ID Variations)', text, re.DOTALL)
    # if dob_section:
    #     dobs = re.findall(r'(\d{2}-\d{2}-\d{4})', dob_section.group(1))
    #     unique_dobs = list(set(dobs))
    #     if len(unique_dobs) > 2:
    #         flags.append({
    #             "flag": "MULTIPLE_DOBS",
    #             "severity": "HIGH",
    #             "detail": f"More than 2 different DOBs reported by different banks: {unique_dobs}",
    #             "values": unique_dobs
    #         })

    return flags

def analyse_enquiries(text):
    enq_section = re.search(r'Inquiries.*?past.*?24.*?months([\s\S]+?)-END OF REPORT', text, re.IGNORECASE)
    if not enq_section:
        return {"total_24m": 0, "recent_6m": 0, "flags": []}

    section = enq_section.group(1)
    enq_dates = re.findall(r'(\d{2}-\d{2}-\d{4})', section)
    total = len(enq_dates)

    today  = date.today()
    recent = 0
    for d in enq_dates:
        try:
            if (today - datetime.strptime(d, "%d-%m-%Y").date()).days <= 180:
                recent += 1
        except: pass

    flags = []
    if recent >= 5:
        flags.append({"flag": "HIGH_RECENT_ENQUIRIES", "severity": "MEDIUM",
            "detail": f"{recent} enquiries in last 6 months — possible loan shopping"})
    if total >= 10:
        flags.append({"flag": "HIGH_TOTAL_ENQUIRIES", "severity": "MEDIUM",
            "detail": f"{total} enquiries in 24 months — very high loan-seeking activity"})

    return {"total_24m": total, "recent_6m": recent, "flags": flags}

def check_freshness(header_info, max_days=365):          # change this 90 days
    doi = header_info.get("date_of_issue")
    if not doi:
        return {"fresh": False, "flag": "DATE_NOT_FOUND",
                "reason": "Date of Issue not found"}

    today = date.today()
    if doi > today:
        return {"fresh": False, "flag": "FUTURE_DATE",
                "reason": f"Date {doi} is in the future — possible tampered document"}

    age = (today - doi).days
    return {
        "fresh":         age <= max_days,
        "date_of_issue": str(doi),
        "age_days":      age,
        "flag":          None if age <= max_days else "REPORT_EXPIRED",
        "reason":        f"Report is {age} days old ({'OK' if age <= max_days else f'EXPIRED — max {max_days} days'})"
    }

def aggregate_risks(accounts, identity_flags, enquiry_result,exposure,delinquency_recency):
    all_flags = list(identity_flags) + enquiry_result.get("flags", [])

    settled, written_off, suit, sub_dbt = [], [], [], []

    for acc in accounts:
        num   = acc["account_number"]
        atype = acc.get("account_type", "Unknown")

        if acc.get("is_settled"):
            amt = acc.get("settlement_amount", 0)
            settled.append(f"Account {num} ({atype}) — settled ₹{amt:,}")

        if acc.get("is_written_off"):
            amt = acc.get("total_writeoff", 0)
            written_off.append(f"Account {num} ({atype}) — written off ₹{amt:,}")

        if acc.get("suit_filed"):
            suit.append(f"Account {num} ({atype})")

        ph = acc.get("payment_history", {})
        if ph.get("worst_severity", 0) >= 2:
            sub_dbt.append(
                f"Account {num} ({atype}) — worst: {ph['worst_status']}, "
                f"bad months: {ph['bad_months_count']}"
            )

    if settled:
        all_flags.append({"flag": "SETTLED_ACCOUNTS", "severity": "HIGH",
            "detail": f"Settled accounts: {settled}", "accounts": settled})

    if written_off:
        all_flags.append({"flag": "WRITTEN_OFF_ACCOUNTS", "severity": "CRITICAL",
            "detail": f"Written-off: {written_off}", "accounts": written_off})

    if suit:
        all_flags.append({"flag": "SUIT_FILED", "severity": "CRITICAL",
            "detail": f"Suit filed on: {suit}", "accounts": suit})

    if sub_dbt:
        sev = "CRITICAL" if any("DBT" in a or "LOS" in a for a in sub_dbt) else "HIGH"
        all_flags.append({"flag": "BAD_PAYMENT_HISTORY", "severity": sev,
            "detail": f"Sub-standard or worse: {sub_dbt}", "accounts": sub_dbt})
        
    # Add to all_flags if flagged
    if exposure.get("flag"):
        all_flags.append(exposure["flag"])

    if delinquency_recency.get("flag"):
        all_flags.append(delinquency_recency["flag"])

    return all_flags

SBFC_MIN_SCORE = 650  # Adjust per SBFC credit policy

def make_decision(score_result, freshness, all_flags, predicted, header):
    critical = [f for f in all_flags if f["severity"] == "CRITICAL"]
    high     = [f for f in all_flags if f["severity"] == "HIGH"]
    medium   = [f for f in all_flags if f["severity"] == "MEDIUM"]

    if not freshness.get("fresh"):
        return "REJECTED", freshness["reason"]

    if not score_result.get("found") or score_result.get("category") == "NO_HISTORY":
        return "MANUAL_REVIEW", "No credit history (NH). Cannot score. Manual underwriting needed."

    # score = score_result["score"]

    if critical:
        names = [f["flag"] for f in critical]
        return "REJECTED", f"Critical flags: {', '.join(names)}"
    
    # ── Resolve the score to use for threshold check ─────────────────────
    # predicted["predicted_score"] is our lender-adjusted score.
    # Fall back to bureau score only if predicted score is unavailable
    # (e.g. calculate_predicted_score returned None for some edge case).
    bureau_score    = score_result.get("score")                       # audit only
    predicted_score = predicted.get("predicted_score") if predicted else None
    effective_score = predicted_score if predicted_score is not None else bureau_score

    if predicted_score is not None and predicted_score != bureau_score:
        adj          = predicted.get("total_adjustment", 0)
        score_note   = (
            f"Bureau score {bureau_score}, "
            f"predicted score {predicted_score} "
            f"({adj:+d} adjustment from behavioural analysis)"
        )
    else:
        score_note = f"Score {effective_score}"
 
    # ── 4. Predicted score below minimum ────────────────────────────────
    if effective_score < SBFC_MIN_SCORE:
        return "REJECTED", (
            f"{score_note}. "
            f"Predicted score {effective_score} below SBFC minimum {SBFC_MIN_SCORE}."
        )

    # if score < SBFC_MIN_SCORE:
    #     return "REJECTED", f"Score {score} below SBFC minimum {SBFC_MIN_SCORE}"

    if high:
        names = [f["flag"] for f in high]
        return "MANUAL_REVIEW", f"Score {score_note} OK but high-severity flags: {', '.join(names)}"

    if medium:
        names = [f["flag"] for f in medium]
        return "APPROVED_WITH_NOTES", f"Score {score_note} OK. Medium flags noted: {', '.join(names)}"
    
    category = predicted.get("category") if predicted else score_result.get("category", "")

    return "APPROVED", f"Score {score_note} ({category}). No flags. Document clean." 

def get_ownership_weight(ownership):
    """
    Reduces penalty based on how much this account is actually
    the applicant's own responsibility.

    INDIVIDUAL / PRIMARY = 100% — they took this loan themselves
    JOINT                = 60%  — shared with co-borrower,
                                   both are liable but not sole owner
    GUARANTOR            = 40%  — only liable if primary defaults,
                                   didn't take the loan themselves

    Example:
    DBT on a JOINT account → penalty -54 X 0.6 = -32 instead of -54
    DBT on a GUARANTOR account → penalty -54 X 0.4 = -22 instead of -54
    """
    weights = {
        "INDIVIDUAL": 1.0,
        "PRIMARY":    1.0,
        "JOINT":      0.6,
        "GUARANTOR":  0.4
    }
    return weights.get((ownership or "").upper(), 1.0)


def calculate_predicted_score(accounts, bureau_score, enquiry_24m=0,utilization=None,vintage=None):
    """
    Lender-side predicted score.
    Starts from bureau score and applies penalties/bonuses
    based on full document analysis.
    
    Returns score + full breakdown of every adjustment made.
    """
    
    if bureau_score is None:
        return {
            "predicted_score": None,
            "category": "NO_HISTORY",
            "adjustments": [],
            "explanation": "No bureau score — cannot calculate predicted score"
        }
    
    score = bureau_score
    adjustments = []
    
    today = date.today()
    
    active_accounts  = [a for a in accounts if a["status"] == "ACTIVE"]
    closed_accounts  = [a for a in accounts if a["status"] == "CLOSED"]
    
    # ── PENALTY 1: Write-off accounts ──────────────────────────────────────
    # Most severe — lender gave up recovering money
    # Penalty reduces based on age (older = slightly less impact)
    for acc in accounts:
        if acc.get("is_written_off") and acc.get("total_writeoff", 0) > 0:
            amt = acc["total_writeoff"]
            weight = get_ownership_weight(acc.get("ownership"))
            
            # Age-based penalty — recent writeoff hurts more
            base_penalty = -60 if amt > 50000 else -45 if amt > 10000 else -35
            penalty      = round(base_penalty * weight)
            
            adjustments.append({
                "factor":      "WRITE_OFF",
                "account":     acc["account_number"],
                 "detail":  f"Write-off ₹{amt:,} on {acc.get('account_type','Unknown')} "
                            f"(ownership: {acc.get('ownership','INDIVIDUAL')}, "
                            f"weight: {weight})",
                "adjustment":  penalty,
                "severity":    "CRITICAL"
            })
            score += penalty
    
    # ── PENALTY 2: Settled accounts ────────────────────────────────────────
    # Paid less than owed — lender took a haircut
    for acc in accounts:
        if acc.get("is_settled") and acc.get("settlement_amount", 0) > 0:
            amt = acc["settlement_amount"]
            weight  = get_ownership_weight(acc.get("ownership"))
            
            base_penalty = -35 if amt > 20000 else -25 if amt > 5000 else -15
            penalty = round(base_penalty * weight)
            
            adjustments.append({
                "factor":     "SETTLEMENT",
                "account":    acc["account_number"],
                "detail":     f"Settled ₹{amt:,} on {acc.get('account_type','Unknown')}",
                "adjustment": penalty,
                "severity":   "HIGH"
            })
            score += penalty
    
    # ── PENALTY 3: DBT/LOS payment history ────────────────────────────────
    # Doubtful or Loss classification = 180+ days overdue at some point
    for acc in accounts:
        ph = acc.get("payment_history", {})
        worst = ph.get("worst_severity", 0)
        bad_months = ph.get("bad_months_count", 0)
        
        if worst >= 3:   # DBT or LOS
            penalty = -40 - (bad_months * 2)   # base + per bad month
            penalty = max(penalty, -70)          # cap at -70
            adjustments.append({
                "factor":     "DBT_LOS_HISTORY",
                "account":    acc["account_number"],
                "detail":     f"Worst status: {ph['worst_status']}, {bad_months} bad months",
                "adjustment": penalty,
                "severity":   "CRITICAL"
            })
            score += penalty
            
        elif worst == 2:  # SUB only
            penalty = -20 - (bad_months * 1)
            penalty = max(penalty, -40)
            adjustments.append({
                "factor":     "SUB_HISTORY",
                "account":    acc["account_number"],
                "detail":     f"Sub-standard history, {bad_months} bad months",
                "adjustment": penalty,
                "severity":   "HIGH"
            })
            score += penalty
    
    # ── PENALTY 4: Currently overdue ──────────────────────────────────────
    total_overdue = sum(a.get("overdue_amount", 0) for a in active_accounts)
    if total_overdue > 0:
        penalty = -50 if total_overdue > 50000 else -35 if total_overdue > 10000 else -20
        adjustments.append({
            "factor":     "CURRENTLY_OVERDUE",
            "detail":     f"Active overdue amount: ₹{total_overdue:,}",
            "adjustment": penalty,
            "severity":   "CRITICAL"
        })
        score += penalty
    
    # ── PENALTY 5: High credit utilisation on active accounts ─────────────
    # Using >80% of sanctioned amount = stressed borrower
    high_util_accounts = []
    for acc in active_accounts:
        if acc.get("disbursed_amount", 0) > 0:
            util = acc["current_balance"] / acc["disbursed_amount"] * 100
            if util > 90:
                high_util_accounts.append((acc["account_number"], round(util,1)))
            elif util > 80:
                high_util_accounts.append((acc["account_number"], round(util,1)))
    
    if high_util_accounts:
        avg_util = sum(u for _,u in high_util_accounts) / len(high_util_accounts)
        penalty = -20 if avg_util > 90 else -10
        adjustments.append({
            "factor":     "HIGH_UTILISATION",
            "detail":     f"High utilisation on accounts: {high_util_accounts}, avg {avg_util:.1f}%",
            "adjustment": penalty,
            "severity":   "MEDIUM"
        })
        score += penalty
    
    # ── PENALTY 6: Too many enquiries ─────────────────────────────────────
    if enquiry_24m >= 10:
        penalty = -15
        adjustments.append({
            "factor":     "HIGH_ENQUIRIES",
            "detail":     f"{enquiry_24m} enquiries in 24 months",
            "adjustment": penalty,
            "severity":   "MEDIUM"
        })
        score += penalty
    elif enquiry_24m >= 5:
        penalty = -8
        adjustments.append({
            "factor":     "MODERATE_ENQUIRIES",
            "detail":     f"{enquiry_24m} enquiries in 24 months",
            "adjustment": penalty,
            "severity":   "LOW"
        })
        score += penalty

    # ── PENALTY 7: Portfolio Credit Utilization ─────────────────────────
    # Portfolio utilization is a stronger indicator than account-level
    # utilization because it measures total borrowing stress.

    if utilization:
        util_pct = utilization.get("utilization_pct", 0)
        if util_pct >= 90:
            adjustments.append({
                "factor": "VERY_HIGH_PORTFOLIO_UTILIZATION",
                "detail": f"Portfolio utilization {util_pct}%",
                "adjustment": -20,
                "severity": "HIGH"
            })
            score -= 20

        elif util_pct >= 75:
            adjustments.append({
                "factor": "HIGH_PORTFOLIO_UTILIZATION",
                "detail": f"Portfolio utilization {util_pct}%",
                "adjustment": -10,
                "severity": "MEDIUM"
            })
            score -= 10
    
    # ── BONUS 1: Clean active accounts ────────────────────────────────────
    clean_active = [
        a for a in active_accounts
        if a.get("payment_history", {}).get("worst_severity", 0) == 0
    ]
    if clean_active:
        bonus = min(len(clean_active) * 8, 25)   # max +25
        adjustments.append({
            "factor":     "CLEAN_ACTIVE_ACCOUNTS",
            "detail":     f"{len(clean_active)} active accounts with perfect payment history",
            "adjustment": +bonus,
            "severity":   "POSITIVE"
        })
        score += bonus
    
    # ── BONUS 2: Zero current overdue ─────────────────────────────────────
    if total_overdue == 0 and len(active_accounts) > 0:
        adjustments.append({
            "factor":     "ZERO_OVERDUE",
            "detail":     "No current overdue on any active account",
            "adjustment": +10,
            "severity":   "POSITIVE"
        })
        score += 10

    # ── BONUS 3: Strong Credit Vintage ──────────────────────────────────
    # Long repayment history generally indicates lower default risk.

    if vintage:
        years = vintage.get("years", 0)
        if years >= 10:
            adjustments.append({
                "factor": "EXCELLENT_CREDIT_VINTAGE",
                "detail": f"{years} years credit history",
                "adjustment": 10,
                "severity": "POSITIVE"
            })
            score += 10
        elif years >= 5:
            adjustments.append({
                "factor": "GOOD_CREDIT_VINTAGE",
                "detail": f"{years} years credit history",
                "adjustment": 5,
                "severity": "POSITIVE"
            })
            score += 5
    
    # Clamp between 300 and 900
    score = max(300, min(900, round(score)))
    
    # Categorise
    if   score >= 800: cat, risk = "EXCELLENT",     "VERY_LOW"
    elif score >= 750: cat, risk = "GOOD",           "LOW"
    elif score >= 650: cat, risk = "AVERAGE",        "MEDIUM"
    elif score >= 550: cat, risk = "BELOW_AVERAGE",  "HIGH"
    else:              cat, risk = "POOR",            "VERY_HIGH"
    
    # Total adjustment
    total_adj = score - bureau_score
    
    return {
        "bureau_score":    bureau_score,
        "predicted_score": score,
        "total_adjustment": total_adj,
        "category":        cat,
        "risk_level":      risk,
        "adjustments":     adjustments,
        "explanation": (
            f"Original bureau score: {bureau_score}. "
            f"Applied {len(adjustments)} lender-side behavioural adjustments "
            f"(settlements, write-offs, repayment history, overdue exposure, "
            f"credit utilisation, enquiries and credit vintage). "
            f"Net adjustment: {total_adj:+d}. "
            f"Final predicted score: {score} ({cat})."
        )
    }


def generate_executive_summary(
    status,
    reason,
    score_analysis,
    risk_summary,
    credit_summary,
    enquiry_analysis,
    applicant,
    foir_readiness
):
    bureau_score = score_analysis.get("score")
    predicted_score = score_analysis.get("predicted", {}).get("predicted_score")

    utilization = (
        credit_summary
        .get("credit_utilization", {})
        .get("utilization_pct")
    )

    vintage = (
        credit_summary
        .get("credit_vintage", {})
        .get("years")
    )

    return {
        "customer": applicant.get("name"),

        "decision": status,

        "risk_level":
            score_analysis.get("predicted", {})
            .get("risk_level"),

        "bureau_score": bureau_score,

        "predicted_score": predicted_score,

        "credit_vintage_years": vintage,

        "portfolio_utilization_pct": utilization,

        "active_loans":
            foir_readiness.get("active_loan_count"),

        "monthly_emi":
            foir_readiness.get("total_monthly_emi"),

        "written_off_accounts":
            risk_summary.get("written_off_count"),

        "settled_accounts":
            risk_summary.get("settled_count"),

        "recent_enquiries":
            enquiry_analysis.get("total_24m"),

        "top_risks":
            risk_summary.get("critical_flags")
            + risk_summary.get("high_flags"),

        "decision_reason": reason
    }

def generate_human_report(result):

    STATUS_EMOJI   = {"APPROVED":"✅","APPROVED_WITH_NOTES":"✅⚠️","MANUAL_REVIEW":"⚠️","REJECTED":"❌"}
    SEVERITY_EMOJI = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🔵","POSITIVE":"🟢"}
    RISK_LABEL     = {"VERY_HIGH":"Very High Risk","HIGH":"High Risk","MEDIUM":"Medium Risk",
                      "LOW":"Low Risk","VERY_LOW":"Very Low Risk","NONE":"No Risk"}
    
    doc_check = result.get("document_type_check", {})

    status    = result.get("overall_status","UNKNOWN")
    applicant = result.get("applicant",{})
    score     = result.get("score_analysis",{})
    predicted = score.get("predicted",{})
    credit    = result.get("credit_summary",{})
    foir      = result.get("foir_readiness",{})
    flags     = result.get("all_flags",[])
    freshness = result.get("freshness_check",{})
    enquiry   = result.get("enquiry_analysis",{})
    accounts  = result.get("account_details",[])
    lines     = []
    add       = lines.append

    # ── HEADER ──────────────────────────────────────────
    add("=" * 60)
    add("  CRIF CREDIT VERIFICATION REPORT — SBFC Finance")
    add("=" * 60)

    add("-" * 60)
    add("  DOCUMENT VALIDATION")
    add("-" * 60)
    add(f"  Document Type : {doc_check.get('detected','Unknown')}")
    add(f"  File Name     : {result.get('file','—')}")
    add(f"  Verified At   : {result.get('verified_at','—')}")
    add("")

    # ── DECISION ────────────────────────────────────────
    add("")
    add(f"  DECISION : {STATUS_EMOJI.get(status,'❓')}  {status}")
    add(f"  REASON   : {result.get('decision_reason','')}")
    add("")

    # ── APPLICANT ────────────────────────────────────────
    add("-" * 60)
    add("  APPLICANT")
    add("-" * 60)
    add(f"  Name           : {applicant.get('name','—')}")
    add(f"  PAN            : {applicant.get('pan','—')}")
    add(f"  Date of Birth  : {applicant.get('dob','—')}")
    add(f"  Application ID : {applicant.get('application_id','—')}")
    add(f"  CRIF Ref       : {applicant.get('chm_ref','—')}")
    add(f"  Report Date    : {freshness.get('date_of_issue','—')}  ({freshness.get('age_days','?')} days old)")
    if applicant.get("multiple_pans"):
        add(f"  ⚠️  Multiple PANs found: {applicant.get('all_pans')}")
    add("")

    # ── SCORE ────────────────────────────────────────────
    add("-" * 60)
    add("  CREDIT SCORE")
    add("-" * 60)
    bureau_score = score.get("score")
    pred_score   = predicted.get("predicted_score")
    if bureau_score:
        add(f"  Bureau Score (CRIF) : {bureau_score}  ({score.get('category','—')})")
        add(f"  Predicted Score     : {pred_score}  ({predicted.get('category','—')}) — {RISK_LABEL.get(predicted.get('risk_level',''),'—')}")
        add(f"  Net Adjustment      : {predicted.get('total_adjustment',0):+d} points")
        add("")
        # add("  What moved the score:")
        # for adj in predicted.get("adjustments",[]):
        #     sev = adj.get("severity","")
        #     if sev == "LOW": continue
        #     e      = SEVERITY_EMOJI.get(sev,"⚪")
        #     factor = adj.get("factor","").replace("_"," ").title()
        #     acc    = f" [Acc {adj['account']}]" if adj.get("account") else ""
        #     add(f"    {e}  {adj['adjustment']:+d}  {factor}{acc}")
        #     add(f"           → {adj.get('detail','')}")
    else:
        add("  Score : No credit history (NH/NA)")
        add("          Refer for manual underwriting.")
    add("")

    # ── CREDIT PROFILE ───────────────────────────────────
    add("-" * 60)
    add("  CREDIT PROFILE")
    add("-" * 60)
    vintage = credit.get("credit_vintage",{})
    util    = credit.get("credit_utilization",{})
    exp     = credit.get("secured_unsecured_exposure",{})
    delinq  = credit.get("delinquency_recency",{})
    add(f"  Credit History Age    : {vintage.get('years','—')} yrs  (oldest: {vintage.get('oldest_account','—')})")
    add(f"  Portfolio Utilisation : {util.get('utilization_pct','—')}%  ({RISK_LABEL.get(util.get('risk',''),'—')})")
    add(f"  Active EMI Burden     : ₹{foir.get('total_monthly_emi',0):,}/month  ({foir.get('active_loan_count',0)} loans)")
    add(f"  Secured Exposure      : ₹{exp.get('secured_balance',0):,}")
    add(f"  Unsecured Exposure    : ₹{exp.get('unsecured_balance',0):,}  ({exp.get('unsecured_pct',0)}% of balance)")
    if delinq.get("has_delinquency"):
        add(f"  Last Bad Payment      : {delinq.get('months_since_last_bad','?')} months ago  (Acc {delinq.get('most_recent_bad_account','?')}) — {RISK_LABEL.get(delinq.get('recency_risk',''),'')}")
    else:
        add("  Last Bad Payment      : None ✅")
    add(f"  Enquiries (24m)       : {enquiry.get('total_24m',0)} total  |  {enquiry.get('recent_6m',0)} in last 6 months")
    add("")

    # ── ACCOUNTS ─────────────────────────────────────────
    add("-" * 60)
    add("  ACCOUNTS")
    add("-" * 60)
    active_accs  = [a for a in accounts if a.get("status") == "ACTIVE"]
    problem_accs = [a for a in accounts if a.get("has_issues")]
    add(f"  Total: {len(accounts)}  |  Active: {len(active_accs)}  |  Problem: {len(problem_accs)}")
    add("")

    if active_accs:
        add("  Active:")
        for acc in active_accs:
            disb = acc.get("disbursed_amount",0)
            bal  = acc.get("current_balance",0)
            u    = f"  Util: {round(bal/disb*100)}%" if disb > 0 else ""
            emi  = f"  EMI: ₹{acc.get('emi_amount',0):,}" if acc.get("emi_amount") else ""
            add(f"    ✅  Acc {acc['account_number']}  {acc.get('account_type','?')} ({acc.get('lender_type','?')})  Bal: ₹{bal:,}{u}{emi}")

    if problem_accs:
        add("")
        add("  Problems:")
        for acc in problem_accs:
            ph = acc.get("payment_history",{})
            add(f"    🔴  Acc {acc['account_number']}  {acc.get('account_type','?')} ({acc.get('lender_type','?')})  Opened: {acc.get('disbursed_date','?')}  Closed: {acc.get('closed_date','?')}")
            if acc.get("is_written_off"):
                add(f"        → Written-off ₹{acc.get('total_writeoff',0):,}")
            if acc.get("is_settled"):
                add(f"        → Settled ₹{acc.get('settlement_amount',0):,} (disbursed ₹{acc.get('disbursed_amount',0):,})")
            if ph.get("worst_severity",0) >= 2:
                add(f"        → Payment history: {ph['worst_status']}  ({ph['bad_months_count']} bad months)")
    add("")

    # ── FLAGS ────────────────────────────────────────────
    add("-" * 60)
    add("  RISK FLAGS")
    add("-" * 60)
    if not flags:
        add("  ✅  No risk flags — document is clean")
    else:
        for sev_label in [("CRITICAL","Critical"),("HIGH","High"),("MEDIUM","Medium")]:
            group = [f for f in flags if f.get("severity") == sev_label[0]]
            for flag in group:
                e    = SEVERITY_EMOJI.get(sev_label[0],"⚪")
                name = flag.get("flag","").replace("_"," ").title()
                add(f"  {e}  {sev_label[1]}: {name}")
                add(f"      {flag.get('detail','')}")
    add("")
    add("=" * 60)
    add("  END OF REPORT")
    add("=" * 60)

    return "\n".join(lines)

from unittest import result


def verify_crif_document(filepath, applicant_pan=None):
    result = {
        "file":            os.path.basename(filepath),
        "verified_at":     datetime.now().isoformat(),
        "overall_status":  None,
        "decision_reason": None,
        "applicant":       {},
        "score_analysis":  {},
        "credit_summary": {},
        "foir_readiness": {},
        "account_summary": {},
        "account_details": [],
        "identity_check":  {},
        "enquiry_analysis":{},
        "freshness_check": {},
        "all_flags":       [],
        "risk_summary":    {},
        "executive_summary": {}
    }

    # File check
    fc = check_file(filepath)
    if not fc["passed"]:
        result["overall_status"]  = "REJECTED"
        result["decision_reason"] = fc["reason"]
        return result

    full_text = extract_full_text(filepath)

    doc_type = detect_document_type(full_text)
    result["document_type_check"] = doc_type
    if doc_type["detected"] != "CRIF":
        result["overall_status"]  = "REJECTED"
        result["decision_reason"] = (
            f"Wrong document uploaded. This appears to be a "
            f"{doc_type['detected']} document, not a CRIF report."
        )
        return result

    # Header
    header = extract_header_info(full_text)
    result["applicant"] = {
        "name":           header.get("applicant_name"),
        "pan":            header.get("pan"),
        "all_pans":       header.get("pans_found"),
        "dob":            header.get("dob"),
        "gender":         header.get("gender"),
        "chm_ref":        header.get("chm_ref"),
        "application_id": header.get("application_id"),
        "multiple_pans":  header.get("multiple_pans", False)
    }

    # # PAN cross-check against applicant-provided PAN
    # if applicant_pan and header.get("pan"):
    #     if header["pan"].upper() != applicant_pan.upper():
    #         result["overall_status"]  = "REJECTED"
    #         result["decision_reason"] = f"PAN mismatch — report: {header['pan']}, applicant: {applicant_pan}"
    #         return result

    # Score
    score_raw = extract_score(full_text)
    if score_raw.get("found") and score_raw.get("score"):
        cat, risk = categorize_score(score_raw["score"])
        score_raw["category"]   = cat
        score_raw["risk_level"] = risk
    result["score_analysis"] = score_raw

    # Accounts
    accounts = extract_all_accounts(full_text)
    result["account_details"] = accounts

    # emi profile
    emi_profile = calculate_emi_profile(accounts)
    result["foir_readiness"] = emi_profile

    vintage = calculate_credit_vintage(accounts)

    utilization = calculate_credit_utilization(accounts)

    exposure = calculate_secured_unsecured_exposure(accounts)

    delinquency_recency = calculate_delinquency_recency(accounts)

    result["credit_summary"] = {
        "credit_vintage": vintage,
        "credit_utilization": utilization,
        "secured_unsecured_exposure": exposure,
        "delinquency_recency": delinquency_recency
    }


    # Identity
    identity_flags = check_identity_consistency(full_text)
    result["identity_check"] = {"flags": identity_flags, "has_issues": len(identity_flags) > 0}

    # Enquiries
    enq = analyse_enquiries(full_text)
    result["enquiry_analysis"] = enq

    # ── Predicted score (lender-side — adjusts bureau score on real behaviour)
    bureau_score  = score_raw.get("score")          # None if NH/NA
    enquiry_24m   = enq.get("total_24m", 0)    # adjust key to your enq dict

    predicted = calculate_predicted_score(
        accounts    = accounts,
        bureau_score = bureau_score,
        enquiry_24m  = enquiry_24m,
        utilization=utilization,
        vintage=vintage
    )
    result["score_analysis"]["predicted"] = predicted


    # Freshness
    freshness = check_freshness(header)
    result["freshness_check"] = freshness

    # Aggregate all flags
    all_flags = aggregate_risks(accounts, identity_flags, enq,exposure,delinquency_recency)
    result["all_flags"] = all_flags

    # Decision
    status, reason = make_decision(score_raw, freshness, all_flags, predicted, header)
    result["overall_status"]  = status
    result["decision_reason"] = reason

    # Risk summary (quick read)
    result["risk_summary"] = {
        "total_flags":            len(all_flags),
        "critical_flags":         [f["flag"] for f in all_flags if f["severity"] == "CRITICAL"],
        "high_flags":             [f["flag"] for f in all_flags if f["severity"] == "HIGH"],
        "medium_flags":           [f["flag"] for f in all_flags if f["severity"] == "MEDIUM"],
        "accounts_with_issues":   [a["account_number"] for a in accounts if a.get("has_issues")],
        "total_accounts_analysed": len(accounts),
        "settled_count":          sum(1 for a in accounts if a.get("is_settled")),
        "written_off_count":      sum(1 for a in accounts if a.get("is_written_off")),
        "predicted_score":        result["score_analysis"]["predicted"]["predicted_score"]
    }

    # Executive Summary
    result["executive_summary"] = generate_executive_summary(
        status=status,
        reason=reason,
        score_analysis=result["score_analysis"],
        risk_summary=result["risk_summary"],
        credit_summary=result["credit_summary"],
        enquiry_analysis=enq,
        applicant=result["applicant"],
        foir_readiness=result["foir_readiness"]
    )

    return result
