"""
Sale Deed Verification Pipeline - Fast LOS Validation.

Phase 1:
    Fast document validation only.

Checks:
    1. File integrity
    2. PDF readability
    3. Page structure
    4. Sale Deed semantic structure
    5. Registration metadata
    6. Party consistency
    7. Financial consistency
    8. Property consistency
    9. Boundary completeness
    10. Representative-page image quality
    11. Blank-page detection
    12. Duplicate-page detection
    13. Tamper analysis

NOT PERFORMED:
    - OCR
    - LLM
    - legal registry verification
    - identity verification
    - Aadhaar verification
    - face detection

Target:
    ~1-2 seconds for normal multi-page PDFs.
"""

from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import pymupdf as fitz
import numpy as np

from src.common.authenticity.tamper import (
    analyze_tampering,
)


class SaleDeedPipeline:

    # ================================================================
    # CONFIG
    # ================================================================

    MAX_FILE_SIZE_MB = 25
    MAX_PAGES = 100

    PREVIEW_DPI = 55
    MAX_REPRESENTATIVE_PAGES = 3

    MIN_PAGE_WIDTH = 300
    MIN_PAGE_HEIGHT = 400

    BLANK_STD_THRESHOLD = 7.0
    DUPLICATE_THRESHOLD = 0.985

    # Strong structural requirements.
    MIN_STRUCTURAL_SIGNALS = 6
    MIN_FINANCIAL_SIGNALS = 2
    MIN_PROPERTY_SIGNALS = 5

    SALE_DEED_KEYWORDS = (
        # English
        "sale deed",
        "sale-deed",
        "deed of sale",
        "vendor",
        "vendee",
        "purchaser",
        "purchasers",
        "seller",
        "buyer",
        "consideration",
        "property",
        "registration",
        "registrar",
        "schedule",
        "conveyance",
        # Hindi / Indian registration terminology
        "विक्रय विलेख",
        "विक्रय",
        "विक्रेता",
        "क्रेता",
        "पक्षकार",
        "संपत्ति",
        "सम्पत्ति",
        "विक्रय मूल्य",
        "विक्रय प्रतिफल",
        "निबंधन",
        "निबंधन कार्यालय",
        "दस्तावेज़ संख्या",
        "दस्तावेज संख्या",
        "खाता संख्या",
        "खेसरा संख्या",
        "प्लाट संख्या",
        "सीमा विवरण",
    )

    REGISTRATION_KEYWORDS = (
        "registration office",
        "registration no",
        "registration number",
        "deed no",
        "deed number",
        "token no",
        "book no",
        "volume no",
        "sub registrar",
        "registered",
        "निबंधन कार्यालय",
        "निबंधन",
        "दस्तावेज़ संख्या",
        "दस्तावेज संख्या",
        "पुस्तक संख्या",
        "खंड संख्या",
        "पृष्ठ संख्या",
        "पंजीकृत",
    )

    PARTY_KEYWORDS = (
        "vendor",
        "seller",
        "buyer",
        "purchaser",
        "purchasers",
        "vendee",
        "executed by",
        "executed",
        "witness",
        "identifier",
        "विक्रेता",
        "क्रेता",
        "पक्षकार",
        "गवाह",
        "निष्पादित",
        "निष्पादक",
        "प्रस्तुतकर्ता",
    )

    PROPERTY_KEYWORDS = (
        "district",
        "sub-district",
        "tehsil",
        "village",
        "khata",
        "khasra",
        "plot",
        "survey",
        "area",
        "boundary",
        "east",
        "west",
        "north",
        "south",
        "जिला",
        "अनुमंडल",
        "तहसील",
        "गांव",
        "गाँव",
        "मौजा",
        "खाता",
        "खेसरा",
        "प्लाट",
        "क्षेत्रफल",
        "सीमा",
        "पूर्व",
        "पश्चिम",
        "उत्तर",
        "दक्षिण",
    )

    # Other common document signals. These are used ONLY to prevent a
    # non-Sale-Deed document from being reported as SALE_DEED.
    OTHER_DOCUMENT_KEYWORDS = {
        "DRIVING_LICENCE": (
            "driving licence", "driving license", "driver license",
            "dl no", "dl number", "licence no", "license no",
            "transport department", "date of birth", "valid till",
        ),
        "PASSPORT": (
            "passport", "republic of india", "nationality",
            "passport no", "passport number",
        ),
        "PAN": (
            "permanent account number", "income tax department",
            "pan card", "pan no", "pan number",
        ),
        "VOTER_ID": (
            "election commission", "elector photo identity",
            "voter id", "epic no", "epic number",
        ),
        "BANK_STATEMENT": (
            "bank statement", "account statement", "account number",
            "opening balance", "closing balance", "transaction date",
        ),
        "SALARY_SLIP": (
            "salary slip", "salary statement", "gross salary",
            "net salary", "basic salary", "earnings", "deductions",
        ),
        "ITR": (
            "income tax return", "itr", "acknowledgement number",
            "assessment year", "gross total income",
        ),
        "CIBIL": (
            "cibil", "credit information report", "credit score",
        ),
        "CRIF": (
            "crif", "credit report", "credit score",
        ),
    }

    FINANCIAL_KEYWORDS = (
        "sale consideration",
        "consideration",
        "market value",
        "stamp duty",
        "registration fee",
        "llr fee",
        "service charge",
        "sale price",
        "विक्रय मूल्य",
        "विक्रय प्रतिफल",
        "मुद्रांक शुल्क",
        "निबंधन शुल्क",
        "अन्य शुल्क",
        "कुल राशि",
    )

    # ================================================================
    # PUBLIC ENTRY
    # ================================================================

    @classmethod
    def verify_document(
        cls,
        file_path: str,
    ) -> dict[str, Any]:

        started = perf_counter()

        path = Path(file_path)

        # ------------------------------------------------------------
        # FILE
        # ------------------------------------------------------------

        if not path.exists():

            return cls._failed(
                "Uploaded Sale Deed does not exist.",
                started,
            )

        if not path.is_file():

            return cls._failed(
                "Uploaded path is not a file.",
                started,
            )

        try:
            file_size = path.stat().st_size
        except OSError:

            return cls._failed(
                "Unable to read uploaded file metadata.",
                started,
            )

        if file_size <= 0:

            return cls._failed(
                "Uploaded file is empty.",
                started,
            )

        if file_size > (
            cls.MAX_FILE_SIZE_MB
            * 1024
            * 1024
        ):

            return cls._failed(
                (
                    f"File exceeds "
                    f"{cls.MAX_FILE_SIZE_MB} MB."
                ),
                started,
            )

        extension = path.suffix.lower()

        if extension == ".pdf":

            return cls._verify_pdf(
                path,
                started,
            )

        if extension in {
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff",
        }:

            return cls._verify_image(
                path,
                started,
            )

        return cls._failed(
            "Unsupported Sale Deed file format.",
            started,
        )

    # ================================================================
    # PDF
    # ================================================================

    @classmethod
    def _verify_pdf(
        cls,
        path: Path,
        started: float,
    ) -> dict[str, Any]:

        try:

            pdf = fitz.open(
                str(path)
            )

        except Exception as exc:

            return cls._failed(
                f"PDF could not be opened: {exc}",
                started,
            )

        try:

            page_count = len(pdf)

            if page_count <= 0:

                return cls._build_response(
                    document_detected=False,
                    page_count=0,
                    image_quality="NOT_CHECKED",
                    tampering_risk="NOT_CHECKED",
                    structural_validation="FAIL",
                    decision="DOCUMENT_REJECTED",
                    score=0,
                    started=started,
                    warnings=[
                        "PDF contains no pages."
                    ],
                    checks={},
                )

            if page_count > cls.MAX_PAGES:

                return cls._build_response(
                    document_detected=False,
                    page_count=page_count,
                    image_quality="NOT_CHECKED",
                    tampering_risk="NOT_CHECKED",
                    structural_validation="FAIL",
                    decision="DOCUMENT_REVIEW",
                    score=30,
                    started=started,
                    warnings=[
                        (
                            f"PDF contains {page_count} pages."
                        )
                    ],
                    checks={},
                )

            # ========================================================
            # FULL TEXT EXTRACTION
            # ========================================================
            #
            # This is NOT OCR.
            #
            # For digitally generated PDFs this is extremely cheap.
            # For scanned PDFs text may be empty, which is allowed.
            # ========================================================

            all_text = cls._extract_pdf_text(
                pdf
            )

            # ========================================================
            # STRUCTURAL PDF CHECK
            # ========================================================

            structural = (
                cls._check_pdf_structure(
                    pdf
                )
            )

            # ========================================================
            # SALE DEED STRUCTURE
            # ========================================================

            semantic = (
                cls._validate_sale_deed_structure(
                    all_text
                )
            )

            document_type, document_confidence = cls._classify_document(
                all_text,
                semantic,
            )

            # ========================================================
            # CROSS-FIELD CONSISTENCY
            # ========================================================

            consistency = (
                cls._validate_consistency(
                    all_text
                )
            )

            # ========================================================
            # REPRESENTATIVE PAGES
            # ========================================================

            page_indexes = (
                cls._representative_pages(
                    page_count
                )
            )

            rendered_pages = []

            for index in page_indexes:

                try:

                    image = cls._render_page(
                        pdf[index]
                    )

                    if image is not None:

                        rendered_pages.append(
                            (
                                index,
                                image,
                            )
                        )

                except Exception:

                    continue

            if not rendered_pages:

                return cls._build_response(
                    document_detected=False,
                    page_count=page_count,
                    image_quality="NOT_CHECKED",
                    tampering_risk="NOT_CHECKED",
                    structural_validation="FAIL",
                    decision="DOCUMENT_REVIEW",
                    score=30,
                    started=started,
                    warnings=[
                        "Representative pages could not be rendered."
                    ],
                    checks={
                        "pdf_structure":
                            structural,
                        "semantic":
                            semantic,
                        "consistency":
                            consistency,
                    },
                )

            # ========================================================
            # IMAGE QUALITY
            # ========================================================

            quality = (
                cls._quality_analysis(
                    rendered_pages
                )
            )

            # ========================================================
            # PAGE INTEGRITY
            # ========================================================

            page_integrity = (
                cls._page_integrity_analysis(
                    rendered_pages
                )
            )

            # ========================================================
            # TAMPERING
            # ========================================================

            tamper = (
                cls._tampering_analysis(
                    rendered_pages
                )
            )

            # ========================================================
            # DOCUMENT DETECTION
            # ========================================================

            if semantic["text_available"]:
                document_detected = (
                    document_type == "SALE_DEED"
                    and semantic["document_detected"]
                )
            else:
                # Without embedded text we cannot reliably classify the
                # document. Do not falsely label it as SALE_DEED.
                document_type = "UNKNOWN"
                document_confidence = 0
                document_detected = False

            # ========================================================
            # FINAL VALIDATION
            # ========================================================

            checks = {

                "pdf_structure":
                    structural["passed"],

                "sale_deed_structure":
                    semantic["passed"],

                "registration_details":
                    semantic["registration_passed"],

                "party_details":
                    semantic["party_passed"],

                "financial_details":
                    semantic["financial_passed"],

                "property_details":
                    semantic["property_passed"],

                "boundary_details":
                    semantic["boundary_passed"],

                "cross_page_consistency":
                    consistency["passed"],

                "page_integrity":
                    not page_integrity[
                        "suspicious"
                    ],

                "tamper_check":
                    tamper["risk"]
                    != "HIGH",

                "image_quality":
                    quality["status"]
                    != "POOR",

            }

            # ========================================================
            # STRONG VALIDATION SCORE
            # ========================================================

            score = cls._calculate_score(
                document_detected=document_detected,
                structural=structural,
                semantic=semantic,
                consistency=consistency,
                quality=quality,
                page_integrity=page_integrity,
                tamper=tamper,
            )

            # ========================================================
            # DECISION
            # ========================================================

            decision = "DOCUMENT_REVIEW"
            review_reasons = []

            if not structural["passed"]:
                review_reasons.append("PDF structure failed.")

            if semantic["text_available"] and document_type != "SALE_DEED":
                if document_type == "UNKNOWN":
                    decision = "DOCUMENT_REVIEW"
                    review_reasons.append(
                        "Uploaded document could not be confidently classified as a Sale Deed."
                    )
                else:
                    decision = "DOCUMENT_REJECTED"
                    review_reasons.append(
                        f"Uploaded document appears to be {document_type}, not a Sale Deed."
                    )

            elif semantic["text_available"] and not semantic["document_detected"]:
                decision = "DOCUMENT_REJECTED"
                review_reasons.append(
                    "Sale Deed identity signals are insufficient."
                )

            elif semantic["text_available"] and not semantic["passed"]:
                decision = "DOCUMENT_REVIEW"
                review_reasons.append(
                    "Required Sale Deed validation sections are incomplete."
                )

            if not consistency["passed"]:
                review_reasons.append("Cross-page consistency failed.")

            if quality["status"] == "POOR":
                review_reasons.append("Representative page quality is poor.")

            if page_integrity["suspicious"]:
                review_reasons.append("Page integrity issue detected.")

            if tamper["risk"] == "HIGH":
                review_reasons.append("High tampering risk detected.")

            # Strong risk signals override a verification decision.
            if tamper["risk"] == "HIGH" or not structural["passed"]:
                decision = "DOCUMENT_REVIEW"

            elif document_type == "SALE_DEED" and semantic["text_available"]:
                if semantic["passed"] and consistency["passed"] and quality["status"] != "POOR" and not page_integrity["suspicious"]:
                    decision = "DOCUMENT_VERIFIED"
                else:
                    decision = "DOCUMENT_REVIEW"

            elif not semantic["text_available"]:
                decision = "DOCUMENT_REVIEW"
                review_reasons.append(
                    "Embedded text unavailable; OCR validation required."
                )

            # ========================================================
            # WARNINGS
            # ========================================================

            warnings = []

            warnings.extend(
                structural.get(
                    "warnings",
                    [],
                )
            )

            warnings.extend(
                semantic.get(
                    "warnings",
                    [],
                )
            )

            warnings.extend(
                consistency.get(
                    "warnings",
                    [],
                )
            )

            warnings.extend(
                page_integrity.get(
                    "warnings",
                    [],
                )
            )

            warnings.extend(
                tamper.get(
                    "warnings",
                    [],
                )
            )

            warnings.extend(
                review_reasons
            )

            # Remove duplicates.
            warnings = list(
                dict.fromkeys(
                    warnings
                )
            )

            return cls._build_response(
                document_detected=document_detected,
                page_count=page_count,
                image_quality=quality["status"],
                tampering_risk=tamper["risk"],
                structural_validation=(
                    "PASS"
                    if structural["passed"]
                    else "FAIL"
                ),
                decision=decision,
                score=score,
                started=started,
                document_type=document_type,
                document_confidence=document_confidence,
                warnings=warnings,
                checks=checks,
            )

        finally:

            pdf.close()

    # ================================================================
    # PDF TEXT
    # ================================================================

    @staticmethod
    def _extract_pdf_text(
        pdf: fitz.Document,
    ) -> str:

        chunks = []

        for page in pdf:

            try:

                text = page.get_text(
                    "text"
                )

                if text:

                    chunks.append(
                        text
                    )

            except Exception:

                continue

        return "\n".join(
            chunks
        ).strip()

    # ================================================================
    # STRUCTURAL PDF CHECK
    # ================================================================

    @classmethod
    def _check_pdf_structure(
        cls,
        pdf: fitz.Document,
    ) -> dict[str, Any]:

        warnings = []

        for index in range(
            len(pdf)
        ):

            try:

                page = pdf[index]

                rect = page.rect

                if (
                    rect.width
                    < cls.MIN_PAGE_WIDTH
                    or
                    rect.height
                    < cls.MIN_PAGE_HEIGHT
                ):

                    warnings.append(
                        (
                            f"Page {index + 1} has "
                            "unusually small dimensions."
                        )
                    )

            except Exception:

                return {
                    "passed": False,
                    "warnings": [
                        (
                            f"Page {index + 1} "
                            "could not be read."
                        )
                    ],
                }

        return {
            "passed": True,
            "warnings": warnings,
        }

    # ================================================================
    # SALE DEED SEMANTIC VALIDATION
    # ================================================================

    @classmethod
    def _validate_sale_deed_structure(
        cls,
        text: str,
    ) -> dict[str, Any]:

        normalized = cls._normalize(
            text
        )

        text_available = (
            len(normalized)
            >= 50
        )

        if not text_available:

            return {

                "text_available":
                    False,

                "document_detected":
                    False,

                "passed":
                    False,

                "registration_passed":
                    False,

                "party_passed":
                    False,

                "financial_passed":
                    False,

                "property_passed":
                    False,

                "boundary_passed":
                    False,

                "matches":
                    0,

                "warnings":
                    [],

            }

        structural_matches = cls._count_keywords(
            normalized,
            cls.SALE_DEED_KEYWORDS,
        )

        registration_matches = cls._count_keywords(
            normalized,
            cls.REGISTRATION_KEYWORDS,
        )

        party_matches = cls._count_keywords(
            normalized,
            cls.PARTY_KEYWORDS,
        )

        financial_matches = cls._count_keywords(
            normalized,
            cls.FINANCIAL_KEYWORDS,
        )

        property_matches = cls._count_keywords(
            normalized,
            cls.PROPERTY_KEYWORDS,
        )

        # ------------------------------------------------------------
        # Registration
        # ------------------------------------------------------------

        registration_passed = (
            registration_matches >= 3
            and
            (
                cls._has_registration_number(
                    normalized
                )
                or
                "deed no" in normalized
            )
        )

        # ------------------------------------------------------------
        # Parties
        # ------------------------------------------------------------

        has_vendor = (
            "vendor" in normalized
            or
            "seller" in normalized
            or
            "executed by" in normalized
            or
            "विक्रेता" in normalized
        )

        has_buyer = (
            "buyer" in normalized
            or
            "purchaser" in normalized
            or
            "vendee" in normalized
            or
            "क्रेता" in normalized
        )

        party_passed = (
            has_vendor
            and
            has_buyer
            and
            party_matches >= 3
        )

        # ------------------------------------------------------------
        # Financial
        # ------------------------------------------------------------

        consideration = (
            cls._extract_money(
                normalized,
                (
                    "sale consideration",
                    "consideration",
                    "विक्रय मूल्य",
                    "विक्रय प्रतिफल",
                ),
            )
        )

        market_value = (
            cls._extract_money(
                normalized,
                (
                    "market value",
                ),
            )
        )

        financial_passed = (
            financial_matches
            >=
            cls.MIN_FINANCIAL_SIGNALS
            and
            (
                consideration is not None
                or
                market_value is not None
            )
        )

        # ------------------------------------------------------------
        # Property
        # ------------------------------------------------------------

        property_passed = (
            property_matches
            >=
            cls.MIN_PROPERTY_SIGNALS
            and
            (
                "area" in normalized
                or
                "क्षेत्रफल" in normalized
            )
        )

        # ------------------------------------------------------------
        # Boundaries
        # ------------------------------------------------------------

        # Boundary labels in real Sale Deeds are often laid out as:
        #
        #     BOUNDARY DESCRIPTION
        #     East : ...
        #     West : ...
        #     North: ...
        #     South: ...
        #
        # Therefore directions are not necessarily adjacent to the word
        # "boundary". Count the direction labels independently.
        boundary_directions = (
            "east",
            "west",
            "north",
            "south",
            "पूर्व",
            "पश्चिम",
            "उत्तर",
            "दक्षिण",
        )

        boundary_count = sum(
            1
            for direction in boundary_directions
            if direction in normalized
        )

        boundary_passed = (
            (
                "boundary" in normalized
                or
                "सीमा" in normalized
            )
            and
            boundary_count >= 3
        )

        # ------------------------------------------------------------
        # Document detection
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # Document detection
        # ------------------------------------------------------------
        #
        # A keyword-count-only detector is too brittle for Indian Sale
        # Deeds because many genuine deeds are bilingual or primarily
        # Hindi. A valid Sale Deed should be identified using independent
        # evidence groups instead of requiring six English keywords.
        #
        # Strong identity signals:
        #   1. Explicit Sale Deed title / Hindi equivalent
        #   2. Seller + buyer party structure
        #   3. Sale consideration / financial evidence
        #   4. Property evidence
        #   5. Registration evidence
        #
        # This remains conservative: a generic registered property
        # document without sale-specific financial/party evidence does
        # not automatically become a Sale Deed.

        explicit_sale_deed_title = (
            "sale deed" in normalized
            or "sale-deed" in normalized
            or "deed of sale" in normalized
            or "sale conveyance" in normalized
            or "conveyance deed" in normalized
            or "विक्रय विलेख" in normalized
            or "विक्रय पत्र" in normalized
        )

        strong_sale_structure = (
            party_passed
            and financial_passed
            and property_passed
        )

        registration_identity = (
            registration_passed
            or "निबंधन" in normalized
            or "registration office" in normalized
            or "registration" in normalized
            or "sub registrar" in normalized
        )

        # A Sale Deed title is a strong identity signal, but the complete
        # validation still requires the individual legal/financial/property
        # checks below. This prevents a random document from passing merely
        # because it contains one generic property keyword.
        document_detected = bool(
            explicit_sale_deed_title
            or (strong_sale_structure and registration_identity)
            or (
                structural_matches >= cls.MIN_STRUCTURAL_SIGNALS
                and registration_identity
                and (party_passed or financial_passed)
            )
        )

        # ------------------------------------------------------------
        # Overall
        # ------------------------------------------------------------

        passed = (
            document_detected
            and
            registration_passed
            and
            party_passed
            and
            financial_passed
            and
            property_passed
            and
            boundary_passed
        )

        warnings = []

        if not registration_passed:

            warnings.append(
                "Registration details are incomplete."
            )

        if not party_passed:

            warnings.append(
                "Vendor/buyer party structure is incomplete."
            )

        if not financial_passed:

            warnings.append(
                "Financial details are incomplete."
            )

        if not property_passed:

            warnings.append(
                "Property identifiers are incomplete."
            )

        if not boundary_passed:

            warnings.append(
                "Property boundary details are incomplete."
            )

        return {

            "text_available":
                True,

            "document_detected":
                document_detected,

            "passed":
                passed,

            "registration_passed":
                registration_passed,

            "party_passed":
                party_passed,

            "financial_passed":
                financial_passed,

            "property_passed":
                property_passed,

            "boundary_passed":
                boundary_passed,

            "matches":
                structural_matches,

            "registration_matches":
                registration_matches,

            "party_matches":
                party_matches,

            "financial_matches":
                financial_matches,

            "property_matches":
                property_matches,

            "boundary_count":
                boundary_count,

            "consideration":
                consideration,

            "market_value":
                market_value,

            "warnings":
                warnings,

        }

    # ================================================================
    # DOCUMENT CLASSIFICATION
    # ================================================================

    @classmethod
    def _classify_document(
        cls,
        text: str,
        semantic: dict[str, Any],
    ) -> tuple[str, int]:
        """Classify the uploaded document before Sale Deed validation.

        Returns (document_type, confidence). The classifier is deliberately
        conservative: a weak generic keyword match is not enough to label a
        document.
        """
        normalized = cls._normalize(text)

        if not normalized:
            return "UNKNOWN", 0

        # Explicit Sale Deed evidence gets priority over generic property
        # terminology.
        sale_title = any(
            term in normalized
            for term in (
                "sale deed",
                "sale-deed",
                "deed of sale",
                "sale conveyance",
                "conveyance deed",
                "विक्रय विलेख",
                "विक्रय पत्र",
            )
        )

        if sale_title:
            return "SALE_DEED", 95

        # Use the existing semantic evidence when no explicit title exists.
        sale_groups = sum(
            bool(semantic.get(key))
            for key in (
                "registration_passed",
                "party_passed",
                "financial_passed",
                "property_passed",
                "boundary_passed",
            )
        )

        if sale_groups >= 4:
            return "SALE_DEED", 85

        candidates: list[tuple[str, int]] = []
        for document_type, keywords in cls.OTHER_DOCUMENT_KEYWORDS.items():
            matches = sum(1 for keyword in keywords if keyword in normalized)
            if matches:
                candidates.append((document_type, matches))

        if candidates:
            candidates.sort(key=lambda item: item[1], reverse=True)
            document_type, matches = candidates[0]

            # Require two independent signals for a hard classification.
            if matches >= 2:
                confidence = min(95, 55 + (matches * 10))
                return document_type, confidence

        return "UNKNOWN", 0

    # ================================================================
    # CONSISTENCY
    # ================================================================

    @classmethod
    def _validate_consistency(
        cls,
        text: str,
    ) -> dict[str, Any]:

        normalized = cls._normalize(
            text
        )

        warnings = []

        # ------------------------------------------------------------
        # Registration number
        # ------------------------------------------------------------

        registration_numbers = re.findall(
            r"(?:registration\s*(?:no|number)|document\s*(?:no|number))\s*\.?\s*[:\-]?\s*([A-Z0-9\/\-]+)",
            normalized,
            flags=re.IGNORECASE,
        )

        registration_numbers = [
            value.upper()
            for value in registration_numbers
        ]

        registration_consistent = True

        if registration_numbers:

            registration_consistent = (
                len(
                    set(
                        registration_numbers
                    )
                )
                ==
                1
            )

            if not registration_consistent:

                warnings.append(
                    "Registration numbers are inconsistent."
                )

        # ------------------------------------------------------------
        # Deed number
        # ------------------------------------------------------------

        deed_numbers = re.findall(
            r"deed\s*(?:no|number)\s*\.?\s*[:\-]?\s*([A-Z0-9\/\-]+)",
            normalized,
            flags=re.IGNORECASE,
        )

        deed_numbers = [
            value.upper()
            for value in deed_numbers
        ]

        deed_consistent = True

        if deed_numbers:

            deed_consistent = (
                len(
                    set(
                        deed_numbers
                    )
                )
                ==
                1
            )

            if not deed_consistent:

                warnings.append(
                    "Deed numbers are inconsistent."
                )

        # ------------------------------------------------------------
        # Financial consistency
        # ------------------------------------------------------------

        financial_consistent = True

        consideration = (
            cls._extract_money(
                normalized,
                (
                    "sale consideration",
                ),
            )
        )

        market_value = (
            cls._extract_money(
                normalized,
                (
                    "market value",
                ),
            )
        )

        if (
            consideration is not None
            and
            market_value is not None
        ):

            # Market value lower than consideration is not inherently
            # impossible in every jurisdiction, so this is only a
            # warning rather than an automatic rejection.
            if market_value < consideration:

                warnings.append(
                    (
                        "Market value is lower than "
                        "sale consideration."
                    )
                )

        # ------------------------------------------------------------
        # Boundary consistency
        # ------------------------------------------------------------

        boundary_consistent = True

        boundary_values = []

        for direction in (
            "east",
            "west",
            "north",
            "south",
        ):

            pattern = (
                rf"boundary\s+{direction}\s*[:\-]?\s*([^\n]+)"
            )

            matches = re.findall(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            )

            if matches:

                boundary_values.append(
                    (
                        direction,
                        matches[-1].strip(),
                    )
                )

        # Duplicate exact boundaries are suspicious.
        values = [
            value
            for _, value
            in boundary_values
        ]

        if (
            len(values) >= 3
            and
            len(set(values))
            <
            len(values)
        ):

            boundary_consistent = False

            warnings.append(
                "Duplicate property boundary values detected."
            )

        passed = (
            registration_consistent
            and
            deed_consistent
            and
            boundary_consistent
        )

        return {

            "passed":
                passed,

            "registration_consistent":
                registration_consistent,

            "deed_consistent":
                deed_consistent,

            "financial_consistent":
                financial_consistent,

            "boundary_consistent":
                boundary_consistent,

            "warnings":
                warnings,

        }

    # ================================================================
    # MONEY
    # ================================================================

    @staticmethod
    def _extract_money(
        text: str,
        labels: tuple[str, ...],
    ) -> int | None:

        for label in labels:

            # Real registration templates often insert parentheses,
            # OCR placeholders or punctuation between the label and the
            # numeric amount, e.g.:
            #   (Sale Consideration) : 2,25,000
            #   (Market Value) : 2,50,000
            # Allow bounded non-digit text instead of requiring the
            # amount to appear immediately after the label.
            pattern = (
                rf"{re.escape(label)}"
                r"[^\d]{0,100}"
                r"(?:rs\.?|₹)?\s*"
                r"([\d,]+)"
            )

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:

                try:

                    return int(
                        match.group(
                            1
                        ).replace(
                            ",",
                            "",
                        )
                    )

                except ValueError:

                    continue

        return None

    # ================================================================
    # REGISTRATION NUMBER
    # ================================================================

    @staticmethod
    def _has_registration_number(
        text: str,
    ) -> bool:

        return bool(
            re.search(
                r"\b[A-Z]{2,5}\/\d{4}\/\d+\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    # ================================================================
    # KEYWORD HELPERS
    # ================================================================

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:

        text = text.lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _count_keywords(
        text: str,
        keywords: tuple[str, ...],
    ) -> int:

        return sum(
            1
            for keyword
            in keywords
            if keyword in text
        )

    # ================================================================
    # REPRESENTATIVE PAGES
    # ================================================================

    @staticmethod
    def _representative_pages(
        page_count: int,
    ) -> list[int]:

        if page_count <= 1:

            return [0]

        if page_count == 2:

            return [
                0,
                1,
            ]

        middle = page_count // 2

        return list(
            dict.fromkeys(
                [
                    0,
                    middle,
                    page_count - 1,
                ]
            )
        )

    # ================================================================
    # RENDER
    # ================================================================

    @classmethod
    def _render_page(
        cls,
        page: fitz.Page,
    ) -> np.ndarray | None:

        scale = (
            cls.PREVIEW_DPI
            /
            72.0
        )

        matrix = fitz.Matrix(
            scale,
            scale,
        )

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        try:

            array = np.frombuffer(
                pix.samples,
                dtype=np.uint8,
            )

            if pix.n == 4:

                image = array.reshape(
                    pix.height,
                    pix.width,
                    4,
                )

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGBA2BGR,
                )

            else:

                image = array.reshape(
                    pix.height,
                    pix.width,
                    3,
                )

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR,
                )

            return image.copy()

        finally:

            del pix

    # ================================================================
    # IMAGE QUALITY
    # ================================================================

    @classmethod
    def _quality_analysis(
        cls,
        rendered_pages: list[
            tuple[int, np.ndarray]
        ],
    ) -> dict[str, Any]:

        scores = []

        for _, image in rendered_pages:

            scores.append(
                cls._quality_score(
                    image
                )
            )

        if not scores:

            return {
                "status":
                    "NOT_CHECKED",
                "score":
                    0.0,
            }

        average = (
            sum(scores)
            /
            len(scores)
        )

        minimum = min(
            scores
        )

        if minimum < 40:

            status = "POOR"

        elif average >= 70:

            status = "GOOD"

        else:

            status = "FAIR"

        return {

            "status":
                status,

            "score":
                round(
                    average,
                    2,
                ),

            "minimum":
                round(
                    minimum,
                    2,
                ),

        }

    @staticmethod
    def _quality_score(
        image: np.ndarray,
    ) -> float:

        if (
            image is None
            or
            image.size == 0
        ):

            return 0.0

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        sharpness = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        brightness = float(
            np.mean(gray)
        )

        contrast = float(
            np.std(gray)
        )

        score = 0.0

        if sharpness >= 100:

            score += 40

        elif sharpness >= 40:

            score += 25

        else:

            score += 10

        if 35 <= brightness <= 235:

            score += 30

        elif 20 <= brightness <= 245:

            score += 20

        else:

            score += 5

        if contrast >= 25:

            score += 30

        elif contrast >= 15:

            score += 20

        else:

            score += 10

        return score

    # ================================================================
    # PAGE INTEGRITY
    # ================================================================

    @classmethod
    def _page_integrity_analysis(
        cls,
        rendered_pages: list[
            tuple[int, np.ndarray]
        ],
    ) -> dict[str, Any]:

        warnings = []

        thumbnails = []

        blank_pages = 0

        for index, image in rendered_pages:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

            std = float(
                np.std(gray)
            )

            if (
                std
                <
                cls.BLANK_STD_THRESHOLD
            ):

                blank_pages += 1

                warnings.append(
                    (
                        f"Representative page "
                        f"{index + 1} appears blank."
                    )
                )

            thumbnail = cv2.resize(
                gray,
                (32, 32),
                interpolation=cv2.INTER_AREA,
            )

            thumbnails.append(
                (
                    index,
                    thumbnail,
                )
            )

        duplicate_found = False

        for i in range(
            len(thumbnails)
        ):

            for j in range(
                i + 1,
                len(thumbnails)
            ):

                similarity = (
                    cls._similarity(
                        thumbnails[i][1],
                        thumbnails[j][1],
                    )
                )

                if (
                    similarity
                    >=
                    cls.DUPLICATE_THRESHOLD
                ):

                    duplicate_found = True

                    warnings.append(
                        (
                            f"Representative pages "
                            f"{thumbnails[i][0] + 1} and "
                            f"{thumbnails[j][0] + 1} "
                            "appear duplicated."
                        )
                    )

        return {

            "suspicious":
                (
                    blank_pages > 0
                    or
                    duplicate_found
                ),

            "blank_pages":
                blank_pages,

            "duplicate_pages":
                duplicate_found,

            "warnings":
                warnings,

        }

    @staticmethod
    def _similarity(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:

        first = (
            first
            .astype(
                np.float32
            )
            .flatten()
        )

        second = (
            second
            .astype(
                np.float32
            )
            .flatten()
        )

        if (
            np.std(first) < 1e-6
            or
            np.std(second) < 1e-6
        ):

            return 0.0

        correlation = np.corrcoef(
            first,
            second,
        )[0, 1]

        if not np.isfinite(
            correlation
        ):

            return 0.0

        return float(
            correlation
        )

    # ================================================================
    # TAMPER
    # ================================================================

    @staticmethod
    def _tampering_analysis(
        rendered_pages: list[
            tuple[int, np.ndarray]
        ],
    ) -> dict[str, Any]:

        risks = []

        warnings = []

        for _, image in rendered_pages:

            try:

                result = (
                    analyze_tampering(
                        image
                    )
                )

            except Exception:

                continue

            if not isinstance(
                result,
                dict,
            ):

                continue

            risk = str(
                result.get(
                    "risk",
                    "UNKNOWN",
                )
            ).upper()

            if risk in {
                "LOW",
                "MEDIUM",
                "HIGH",
            }:

                risks.append(
                    risk
                )

        if "HIGH" in risks:

            warnings.append(
                "High tampering signal detected."
            )

            return {
                "risk":
                    "HIGH",
                "warnings":
                    warnings,
            }

        if "MEDIUM" in risks:

            warnings.append(
                "Moderate tampering signal detected."
            )

            return {
                "risk":
                    "MEDIUM",
                "warnings":
                    warnings,
            }

        if "LOW" in risks:

            return {
                "risk":
                    "LOW",
                "warnings":
                    [],
            }

        return {
            "risk":
                "NOT_CHECKED",
            "warnings":
                [],
        }

    # ================================================================
    # SCORE
    # ================================================================

    @staticmethod
    def _calculate_score(
        document_detected: bool,
        structural: dict[str, Any],
        semantic: dict[str, Any],
        consistency: dict[str, Any],
        quality: dict[str, Any],
        page_integrity: dict[str, Any],
        tamper: dict[str, Any],
    ) -> int:

        if not document_detected:
            return 0

        score = 0

        # ------------------------------------------------------------
        # PDF
        # ------------------------------------------------------------

        if structural["passed"]:

            score += 15

        # ------------------------------------------------------------
        # Document
        # ------------------------------------------------------------

        if document_detected:

            score += 15

        # ------------------------------------------------------------
        # Semantic validation
        # ------------------------------------------------------------

        if semantic.get(
            "registration_passed"
        ):

            score += 10

        if semantic.get(
            "party_passed"
        ):

            score += 10

        if semantic.get(
            "financial_passed"
        ):

            score += 10

        if semantic.get(
            "property_passed"
        ):

            score += 10

        if semantic.get(
            "boundary_passed"
        ):

            score += 5

        # ------------------------------------------------------------
        # Consistency
        # ------------------------------------------------------------

        if consistency["passed"]:

            score += 10

        # ------------------------------------------------------------
        # Quality
        # ------------------------------------------------------------

        if quality["status"] == "GOOD":

            score += 5

        elif quality["status"] == "FAIR":

            score += 3

        # ------------------------------------------------------------
        # Page integrity
        # ------------------------------------------------------------

        if not page_integrity["suspicious"]:

            score += 5

        # ------------------------------------------------------------
        # Tamper
        # ------------------------------------------------------------

        if tamper["risk"] == "LOW":

            score += 5

        elif tamper["risk"] == "MEDIUM":

            score += 2

        return max(
            0,
            min(
                100,
                score,
            ),
        )

    # ================================================================
    # IMAGE FILE
    # ================================================================

    @classmethod
    def _verify_image(
        cls,
        path: Path,
        started: float,
    ) -> dict[str, Any]:

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            return cls._failed(
                "Image could not be decoded.",
                started,
            )

        quality = cls._single_quality(
            image
        )

        tamper = "NOT_CHECKED"

        try:

            result = (
                analyze_tampering(
                    image
                )
            )

            if isinstance(
                result,
                dict,
            ):

                tamper = str(
                    result.get(
                        "risk",
                        "NOT_CHECKED",
                    )
                ).upper()

        except Exception:

            pass

        decision = (
            "DOCUMENT_REVIEW"
        )

        # Image-only documents cannot be classified reliably without OCR.
        # Do not hardcode SALE_DEED here.
        if (
            quality == "POOR"
            or
            tamper == "HIGH"
        ):

            decision = (
                "DOCUMENT_REVIEW"
            )

        return cls._build_response(
            document_detected=True,
            page_count=1,
            image_quality=quality,
            tampering_risk=tamper,
            structural_validation="PASS",
            decision=decision,
            score=50,
            started=started,
            document_type="UNKNOWN",
            document_confidence=0,
            warnings=[
                (
                    "Image-only Sale Deed requires "
                    "OCR validation."
                )
            ],
            checks={},
        )

    @classmethod
    def _single_quality(
        cls,
        image: np.ndarray,
    ) -> str:

        score = cls._quality_score(
            image
        )

        if score >= 70:

            return "GOOD"

        if score >= 40:

            return "FAIR"

        return "POOR"

    # ================================================================
    # RESPONSE
    # ================================================================

    @classmethod
    def _build_response(
        cls,
        document_detected: bool,
        page_count: int,
        image_quality: str,
        tampering_risk: str,
        structural_validation: str,
        decision: str,
        score: int,
        started: float,
        warnings: list[str] | None = None,
        checks: dict[str, Any] | None = None,
        document_type: str = "UNKNOWN",
        document_confidence: int = 0,
    ) -> dict[str, Any]:

        elapsed = round(
            perf_counter()
            -
            started,
            3,
        )

        return {

            "document_type":
                document_type,

            "document_confidence":
                document_confidence,

            "decision":
                decision,

            "score":
                score,

            "validation": {

                "document_detected":
                    document_detected,

                "image_quality":
                    image_quality,

                "tampering_risk":
                    tampering_risk,

                "structural_validation":
                    structural_validation,

            },

            "los": {

                "document_validation":
                    decision,

                "requires_ocr_phase":
                    decision
                    !=
                    "DOCUMENT_VERIFIED",

                "requires_rcu_review":
                    decision
                    ==
                    "DOCUMENT_REVIEW",

            },

            "processing_time_seconds":
                elapsed,

            "within_target":
                elapsed <= 2.0,

            # Keep detailed diagnostics available
            # without polluting the normal response.
            "checks":
                checks or {},

            "warnings":
                list(
                    dict.fromkeys(
                        warnings or []
                    )
                ),

        }

    # ================================================================
    # FAILED
    # ================================================================

    @classmethod
    def _failed(
        cls,
        reason: str,
        started: float,
    ) -> dict[str, Any]:

        return cls._build_response(

            document_detected=False,

            page_count=0,

            image_quality="NOT_CHECKED",

            tampering_risk="NOT_CHECKED",

            structural_validation="FAIL",

            decision="DOCUMENT_REJECTED",

            score=0,

            started=started,

            warnings=[
                reason
            ],

            checks={},

        )


# --------------------------------------------------------------------
# OPTIONAL BACKWARD-COMPATIBLE ALIAS
# --------------------------------------------------------------------

VerificationPipeline = SaleDeedPipeline