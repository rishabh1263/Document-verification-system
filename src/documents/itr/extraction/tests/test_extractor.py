"""
Tests for ITR field extraction.

Current extraction scope:
1. Name
2. PAN
3. Assessment Year
4. Date of Birth
5. Total Income
6. Business / Profession Income
"""

from __future__ import annotations

from pathlib import Path

import fitz

from src.documents.itr.extraction.extractor import (
    extract_name,
    extract_pan,
    extract_assessment_year,
    extract_dob,
    extract_total_income,
    extract_business_income,
)


PDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "Vedant ITR.pdf"
)

SHASHIKANT_PDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "Shashiant ITR.pdf"
)


def _read_pdf_text(pdf_path: Path) -> str:
    """Read all native PDF text."""

    assert pdf_path.exists(), (
        f"Test PDF not found: {pdf_path}"
    )

    document = fitz.open(str(pdf_path))

    try:
        text = "\n".join(
            page.get_text("text")
            for page in document
        )
    finally:
        document.close()

    assert text.strip()

    return text


# ==========================================================
# NAME TESTS
# ==========================================================

def test_extract_name_colon_format() -> None:
    text = """
    Assessment Year: 2024-25
    Name: Rishabh Singh
    Address: Mumbai, Maharashtra
    """
    assert extract_name(text) == "Rishabh Singh"


def test_extract_name_hyphen_format() -> None:
    text = """
    Assessment Year: 2024-25
    Name - Rishabh Singh
    Address - Mumbai, Maharashtra
    """
    assert extract_name(text) == "Rishabh Singh"


def test_extract_name_next_line_format() -> None:
    text = """
    Assessment Year
    2024-25

    Name
    Rishabh Singh

    Address
    Mumbai, Maharashtra
    """
    assert extract_name(text) == "Rishabh Singh"


def test_extract_name_of_assessee() -> None:
    text = """
    Assessment Year: 2024-25
    Name of Assessee: Rishabh Singh
    Address: Mumbai, Maharashtra
    """
    assert extract_name(text) == "Rishabh Singh"


def test_extract_name_missing_returns_none() -> None:
    text = """
    Assessment Year: 2024-25
    PAN: ABCDE1234F
    Address: Mumbai, Maharashtra
    Status: Individual
    """
    assert extract_name(text) is None


def test_extract_name_empty_text_returns_none() -> None:
    assert extract_name("") is None
    assert extract_name("   ") is None
    assert extract_name(None) is None


def test_extract_name_from_real_vedant_itr() -> None:
    text = _read_pdf_text(PDF_PATH)
    assert extract_name(text) == "Vedant Ashish Sinagare"


# ==========================================================
# PAN TESTS
# ==========================================================

def test_extract_pan_colon_format() -> None:
    text = """
    Assessment Year: 2024-25
    PAN: ABCDE1234F
    Name: Rishabh Singh
    """
    assert extract_pan(text) == "ABCDE1234F"


def test_extract_pan_hyphen_format() -> None:
    text = "PAN - ABCDE1234F"
    assert extract_pan(text) == "ABCDE1234F"


def test_extract_pan_standalone() -> None:
    text = """
    Permanent Account Number
    ABCDE1234F
    """
    assert extract_pan(text) == "ABCDE1234F"


def test_extract_pan_lowercase_is_normalized() -> None:
    assert extract_pan("PAN: abcde1234f") == "ABCDE1234F"


def test_extract_pan_ocr_spaced_format() -> None:
    text = """
    PAN:
    A B C D E 1 2 3 4 F
    """
    assert extract_pan(text) == "ABCDE1234F"


def test_extract_pan_missing_returns_none() -> None:
    text = """
    Assessment Year: 2024-25
    Name: Rishabh Singh
    Address: Mumbai
    """
    assert extract_pan(text) is None


def test_extract_pan_invalid_format_returns_none() -> None:
    assert extract_pan("PAN: ABC12345F") is None


def test_extract_pan_empty_text_returns_none() -> None:
    assert extract_pan("") is None
    assert extract_pan("   ") is None
    assert extract_pan(None) is None


def test_extract_pan_from_real_vedant_itr() -> None:
    text = _read_pdf_text(PDF_PATH)
    assert extract_pan(text) == "MCVPS7350E"


# ==========================================================
# ASSESSMENT YEAR TESTS
# ==========================================================

def test_extract_assessment_year_colon_format() -> None:
    assert (
        extract_assessment_year(
            "Assessment Year: 2024-25\nName: Rishabh Singh"
        )
        == "2024-25"
    )


def test_extract_assessment_year_without_colon() -> None:
    assert (
        extract_assessment_year(
            "Assessment Year 2024-25\nName: Rishabh Singh"
        )
        == "2024-25"
    )


def test_extract_assessment_year_ay_format() -> None:
    assert (
        extract_assessment_year(
            "AY: 2024-25\nPAN: ABCDE1234F"
        )
        == "2024-25"
    )


def test_extract_assessment_year_ay_dot_format() -> None:
    assert (
        extract_assessment_year(
            "A.Y. 2024-25\nPAN: ABCDE1234F"
        )
        == "2024-25"
    )


def test_extract_assessment_year_next_line() -> None:
    text = """
    Assessment Year
    2024-25

    Name
    Rishabh Singh
    """
    assert extract_assessment_year(text) == "2024-25"


def test_extract_assessment_year_slash_format() -> None:
    assert (
        extract_assessment_year(
            "Assessment Year: 2024/25"
        )
        == "2024-25"
    )


def test_extract_assessment_year_missing_returns_none() -> None:
    text = """
    PAN: ABCDE1234F
    Name: Rishabh Singh
    """
    assert extract_assessment_year(text) is None


def test_extract_assessment_year_empty_returns_none() -> None:
    assert extract_assessment_year("") is None
    assert extract_assessment_year("   ") is None
    assert extract_assessment_year(None) is None


def test_extract_assessment_year_from_real_vedant_itr() -> None:
    text = _read_pdf_text(PDF_PATH)
    assert extract_assessment_year(text) == "2024-25"


# ==========================================================
# DOB TESTS
# ==========================================================

def test_extract_dob_colon_format() -> None:
    assert (
        extract_dob(
            "Name: Rishabh Singh\nDOB: 04/01/2001"
        )
        == "04/01/2001"
    )


def test_extract_dob_hyphen_format() -> None:
    assert (
        extract_dob(
            "Name: Rishabh Singh\n"
            "Date of Birth - 04-01-2001"
        )
        == "04/01/2001"
    )


def test_extract_dob_next_line_format() -> None:
    assert (
        extract_dob(
            "Name: Rishabh Singh\n"
            "Date of Birth\n"
            "04/01/2001"
        )
        == "04/01/2001"
    )


def test_extract_dob_invalid_returns_none() -> None:
    assert (
        extract_dob(
            "Name: Rishabh Singh\nDOB: 45/19/2001"
        )
        is None
    )


def test_extract_dob_from_real_vedant_itr() -> None:
    text = _read_pdf_text(PDF_PATH)
    assert extract_dob(text) == "04/01/2001"


def test_extract_dob_from_shashikant_itr() -> None:
    text = _read_pdf_text(
        SHASHIKANT_PDF_PATH
    )
    assert extract_dob(text) == "22/02/1972"


# ==========================================================
# TOTAL INCOME TESTS
# ==========================================================

def test_extract_total_income_colon_format() -> None:
    assert (
        extract_total_income(
            "Name: Rishabh Singh\n"
            "Total Income: 498000"
        )
        == 498000
    )


def test_extract_total_income_comma_format() -> None:
    assert (
        extract_total_income(
            "Name: Rishabh Singh\n"
            "Total Income: 4,98,000"
        )
        == 498000
    )


def test_extract_total_income_table_format() -> None:
    text = """
    Income chargeable under the head "Business and Profession"
    4,98,000
    ■ Total Income
    4,98,000
    Tax on total income
    9,900
    """
    assert extract_total_income(text) == 498000


def test_extract_total_income_ignores_updated_return() -> None:
    text = """
    Total Income as per Updated return
    3,50,000

    Total Income as per earlier return
    4,00,000

    ■ Total Income
    4,98,000
    """
    assert extract_total_income(text) == 498000


def test_extract_total_income_from_real_vedant_itr() -> None:
    text = _read_pdf_text(PDF_PATH)
    assert extract_total_income(text) == 498000


# ==========================================================
# BUSINESS / PROFESSION INCOME TESTS
# ==========================================================

def test_extract_business_income_colon_format() -> None:
    text = """
    Business: Presumptive profits u/s 44AD
    4,98,000
    """
    assert extract_business_income(text) == 498000


def test_extract_business_income_table_format() -> None:
    text = """
    ■ Profits and gains of Business or Profession
    Business: Presumptive profits u/s 44AD
    1
    4,98,000
    Income chargeable under the head "Business and Profession"
    4,98,000
    """
    assert extract_business_income(text) == 498000


def test_extract_business_income_ignores_table_serial_number() -> None:
    text = """
    Business: Presumptive profits u/s 44AD
    1
    4,98,000
    """

    assert (
        extract_business_income(text)
        == 498000
    )


def test_extract_business_income_ungrouped_amount() -> None:
    text = """
    Business: Profession
    498000
    """

    assert (
        extract_business_income(text)
        == 498000
    )


def test_extract_business_income_small_valid_amount() -> None:
    text = """
    Business: Profession
    500
    """

    assert (
        extract_business_income(text)
        == 500
    )


def test_extract_business_income_ignores_tax_amount() -> None:
    text = """
    Business: Presumptive profits u/s 44AD
    4,98,000
    Tax on total income
    9,900
    Rebate u/s 87A
    9,900
    """
    assert extract_business_income(text) == 498000


def test_extract_business_income_missing_returns_none() -> None:
    text = """
    Name: Rishabh Singh
    Total Income: 498000
    """
    assert extract_business_income(text) is None


def test_extract_business_income_empty_returns_none() -> None:
    assert extract_business_income("") is None
    assert extract_business_income("   ") is None
    assert extract_business_income(None) is None


def test_extract_business_income_from_real_vedant_itr() -> None:
    text = _read_pdf_text(PDF_PATH)
    assert extract_business_income(text) == 498000