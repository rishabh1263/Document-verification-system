"""
Regex Extractor â€” Production Ready

Smart financial extraction: finds the financial paragraph first, then extracts numbers.
"""

import re


class RegexExtractor:

    def __init__(self):
        self.patterns = self._compile_patterns()

    def _compile_patterns(self):
        return {
            # -- Document Details --
            "deed_number": [
                r"Deed\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
                r"Document\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
                r"à¤¦à¤¸à¥à¤¤à¤¾à¤µà¥‡à¤œà¤¼\s*à¤¸à¤‚à¤–à¥à¤¯à¤¾[\s:\-]*([A-Za-z0-9\-/]+)",
            ],
            "token_number": [
                r"Token\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
                r"Token\s*Number\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
                r"à¤Ÿà¥‹à¤•à¤¨\s*à¤¨à¤‚[\s:\-]*([A-Za-z0-9\-/]+)",
            ],
            "registration_number": [
                r"Registration\s*Number\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
                r"Reg\.?\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
                r"à¤ªà¤‚à¤œà¥€à¤•à¤°à¤£\s*à¤¸à¤‚à¤–à¥à¤¯à¤¾[\s:\-]*([A-Za-z0-9\-/]+)",
                r"à¤ªà¤‚\.\s*à¤¸à¤‚\.?\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
            ],
            "serial_number": [
                r"Serial\s*Number\s*[:\-]?\s*([0-9]+)",
                r"Serial\s*No\.?\s*[:\-]?\s*([0-9]+)",
                r"S\.No\.?\s*[:\-]?\s*([0-9]+)",
                r"à¤•à¥à¤°à¤®\s*à¤¸à¤‚à¤–à¥à¤¯à¤¾[\s:\-]*([0-9]+)",
            ],
            "registration_date": [
                r"Registration\s*Date\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
                r"à¤¦à¤¿à¤¨à¤¾à¤‚à¤•[\s:\-]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
                r"Date\s*[:\-]?\s*([0-9]{2}/[0-9]{2}/[0-9]{4})",
            ],
            "registration_office": [
                r"Registration\s*Office\s*,?\s*([A-Za-z\s]+?)(?:\n|Sub\s*Registrar|District)",
                r"Office\s*of\s*the\s*Sub\s*Registrar\s*,?\s*([A-Za-z\s]+?)(?:\n|District)",
                r"à¤ªà¤‚à¤œà¥€à¤•à¤°à¤£\s*à¤•à¤¾à¤°à¥à¤¯à¤¾à¤²à¤¯[\s:\-]*([\u0900-\u097F\s]+?)(?:\n|à¤œà¤¿à¤²à¤¾)",
            ],
            "book_number": [
                r"Book\s*No\.?\s*[:\-]?\s*([A-Za-z0-9]+)",
                r"Book\s*Number\s*[:\-]?\s*([A-Za-z0-9]+)",
                r"à¤¬à¥à¤•\s*à¤¨à¤‚[\s:\-]*([A-Za-z0-9]+)",
            ],
            "volume_number": [
                r"Volume\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-]+)",
                r"Volume\s*Number\s*[:\-]?\s*([A-Za-z0-9\-]+)",
                r"à¤µà¥‰à¤²à¥à¤¯à¥‚à¤®\s*à¤¨à¤‚[\s:\-]*([A-Za-z0-9\-]+)",
            ],
            "page_number": [
                r"Page\s*No\.?\s*[:\-]?\s*([0-9\-\s]+)",
                r"Page\s*Number\s*[:\-]?\s*([0-9\-\s]+)",
                r"à¤ªà¥ƒà¤·à¥à¤ \s*à¤¸à¤‚à¤–à¥à¤¯à¤¾[\s:\-]*([0-9\-\s]+)",
            ],

            # -- Property --
            "district": [
                r"District[\s:\-]*([A-Za-z\s]+?)(?:\n|Village|Tehsil|Taluka|Sub[-\s]?District)",
                r"à¤œà¤¿à¤²à¤¾[\s:\-]*([\u0900-\u097F\s]+?)(?:\n|à¤—à¤¾à¤à¤µ|à¤—à¤¾à¤‚à¤µ|à¤¤à¤¹à¤¸à¥€à¤²|à¤¤à¤¾à¤²à¥à¤•à¤¾)",
            ],
            "village": [
                r"Village[\s:\-]*([A-Za-z\s]+?)(?:\n|District|Tehsil|Taluka|Survey|Plot|Khasra|Khata)",
                r"(?:à¤—à¤¾à¤à¤µ|à¤—à¤¾à¤‚à¤µ)[\s:\-]*([\u0900-\u097F\s]+?)(?:\n|à¤œà¤¿à¤²à¤¾|à¤¤à¤¹à¤¸à¥€à¤²|à¤¤à¤¾à¤²à¥à¤•à¤¾|à¤¸à¤°à¥à¤µà¥‡|à¤ªà¥à¤²à¥‰à¤Ÿ|à¤–à¤¸à¤°à¤¾|à¤–à¤¾à¤¤à¤¾)",
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
                r"(?:Area|Land\s*Area)[\s:\-]*([0-9\.,]+\s*(?:sq\.?\s*ft|sq\.?\s*yd|acre|bigha|kattha|decimal|hectare|ha))",
                r"(?:à¤•à¥à¤·à¥‡à¤¤à¥à¤°à¤«à¤²|à¤•à¥à¤·à¥‡à¤¤à¥à¤°)[\s:\-]*([0-9\.,]+\s*(?:à¤µà¤°à¥à¤—|à¤µà¤°à¥à¤—à¤«à¥à¤Ÿ|à¤µà¤°à¥à¤—à¤®à¥€à¤Ÿà¤°|à¤à¤•à¤¡à¤¼|à¤¬à¥€à¤˜à¤¾|à¤•à¤Ÿà¥à¤ à¤¾|à¤¡à¥‡à¤¸à¥€à¤®à¤²|à¤¹à¥‡à¤•à¥à¤Ÿà¥‡à¤¯à¤°))",
            ],
            "tehsil": [
                r"(?:Tehsil|Taluka)[\s:\-]*([A-Za-z\s]+?)(?:\n|District|Village|Sub[-\s]?District)",
                r"(?:à¤¤à¤¹à¤¸à¥€à¤²|à¤¤à¤¾à¤²à¥à¤•à¤¾)[\s:\-]*([\u0900-\u097F\s]+?)(?:\n|à¤œà¤¿à¤²à¤¾|à¤—à¤¾à¤à¤µ|à¤—à¤¾à¤‚à¤µ)",
            ],
        }

    # =========================================================
    # SMART FINANCIAL EXTRACTION
    # =========================================================

    def _extract_smart_financial(self, text: str):
        """
        Smart financial extraction:
        1. Find financial paragraph using OCR-friendly keywords
        2. Extract all 3+ digit numbers from context
        3. Assign: first = stamp_duty, second = registration_fee
        """
        result = {
            "stamp_duty": "",
            "registration_fee": "",
            "sale_consideration": "",
            "market_value": "",
            "other_fee": ""
        }

        # Step 1: Find financial paragraph using OCR-friendly keywords
        financial_keywords = [
            r"à¤®à¥à¤¦à¥à¤°",           # part of à¤®à¥à¤¦à¥à¤°à¤¾à¤‚à¤•/à¤®à¥à¤¦à¥à¤°à¤‚à¤• (stamp)
            r"Stamp",
            r"stamp\s+duty",
            r"à¤¶à¥à¤²à¥à¤•",           # fee/duty
            r"registration\s+fee",
            r"à¤ªà¤‚à¤œà¥€à¤•à¤°à¤£\s+à¤¶à¥à¤²à¥à¤•",
        ]

        context = ""
        start_pos = -1

        for keyword in financial_keywords:
            match = re.search(keyword, text, re.IGNORECASE)
            if match:
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 300)
                candidate = text[start:end]
                # Pick the longest context (most likely to have both numbers)
                if len(candidate) > len(context):
                    context = candidate
                    start_pos = start

        if not context:
            return result

        print(f"[Regex] Financial context found: {context[:200]}...")

        # Step 2: Extract all numbers with 3+ digits
        numbers = re.findall(r"\d{3,}", context)

        if not numbers:
            return result

        print(f"[Regex] Numbers found in context: {numbers}")

        # Step 3: Assign values
        # First number = stamp duty, second = registration fee
        if len(numbers) >= 1:
            result["stamp_duty"] = "Rs. " + numbers[0]
        if len(numbers) >= 2:
            result["registration_fee"] = "Rs. " + numbers[1]
        if len(numbers) >= 3:
            # Third could be sale consideration or other fee
            result["other_fee"] = "Rs. " + numbers[2]

        return result

    # =========================================================
    # MAIN EXTRACT
    # =========================================================

    def extract(self, text: str):
        result = {
            "document_details": {
                "deed_number": "",
                "token_number": "",
                "registration_number": "",
                "serial_number": "",
                "registration_date": "",
                "registration_office": "",
                "book_number": "",
                "volume_number": "",
                "page_number": "",
                "document_type": ""
            },
            "financial": {
                "stamp_duty": "",
                "registration_fee": "",
                "sale_consideration": "",
                "market_value": "",
                "other_fee": ""
            },
            "property": {
                "district": "",
                "village": "",
                "survey_number": "",
                "plot_number": "",
                "khasra_number": "",
                "khata_number": "",
                "area": ""
            }
        }

        # --- Smart Financial Extraction (PRIORITY) ---
        smart_financial = self._extract_smart_financial(text)
        result["financial"].update(smart_financial)

        # --- Document Details & Property (standard regex) ---
        match_counts = {"document_details": 0, "property": 0}
        total_fields = {"document_details": 10, "property": 8}

        for field_name, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    clean_match = self._clean_value(matches[0])
                    if clean_match and self._is_valid_value(field_name, clean_match):
                        if field_name in result["document_details"]:
                            result["document_details"][field_name] = clean_match
                            match_counts["document_details"] += 1
                        elif field_name in result["property"]:
                            result["property"][field_name] = clean_match
                            match_counts["property"] += 1
                        break

        # Add per-section confidence
        for section in ["document_details", "property"]:
            filled = match_counts[section]
            total = total_fields[section]
            result[section]["_confidence"] = round((filled / total) * 100, 1) if total > 0 else 0

        # Financial confidence
        fin_filled = sum(1 for v in result["financial"].values() if v and not v.startswith("Rs. "))
        fin_filled += sum(1 for k, v in result["financial"].items() if v and k in ["stamp_duty", "registration_fee"])
        result["financial"]["_confidence"] = round((fin_filled / 5) * 100, 1)

        return result

    def _clean_value(self, value):
        if isinstance(value, tuple):
            value = value[0] if value else ""
        if not value:
            return ""
        value = value.strip()
        value = re.sub(r"[;:,]$", "", value)
        return value.strip()

    def _is_valid_value(self, field_name, value):
        if not value or len(value) < 2:
            return False
        return True

    def extract_sale_consideration_candidates(self, text: str):
        """Legacy method â€” kept for compatibility."""
        candidates = []
        # Try smart financial first
        smart = self._extract_smart_financial(text)
        if smart.get("stamp_duty"):
            candidates.append({"raw": smart["stamp_duty"], "normalized": None})
        if smart.get("registration_fee"):
            candidates.append({"raw": smart["registration_fee"], "normalized": None})
        return candidates
