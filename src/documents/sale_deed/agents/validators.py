class ValidationAgent:

    def validate(self, extracted_data: dict) -> dict:
        document = extracted_data.get("document_details", {})
        parties = extracted_data.get("party_roles", {})
        property_data = extracted_data.get("property", {})
        financial = extracted_data.get("financial", {})

        passed_checks = []
        critical_missing = []
        optional_missing = []
        warnings = []

        # Document validation
        self._validate_document(
            document,
            passed_checks,
            critical_missing
        )

        # Party validation
        self._validate_parties(
            parties,
            passed_checks,
            critical_missing
        )

        # Property validation
        self._validate_property(
            property_data,
            passed_checks,
            optional_missing
        )

        # Financial validation
        self._validate_financial(
            financial,
            passed_checks,
            warnings
        )

        completeness = self._calculate_completeness(
            passed_checks,
            critical_missing,
            optional_missing
        )

        confidence = self._calculate_confidence(
            document,
            parties,
            property_data,
            financial
        )

        status = (
            "Verified"
            if not critical_missing
            else "Needs Review"
        )

        return {
            "status": status,
            "confidence": confidence,
            "completeness": completeness,
            "passed_checks": passed_checks,
            "critical_missing": critical_missing,
            "optional_missing": optional_missing,
            "warnings": warnings,
            "manual_review": len(critical_missing) > 0
        }
