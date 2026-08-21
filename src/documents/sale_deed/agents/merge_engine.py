from copy import deepcopy


class MergeEngine:

    # Fields that Regex is more reliable for
    REGEX_PRIORITY = {
        "document_details": [
            "deed_number",
            "token_number",
            "registration_number",
            "registration_date",
            "registration_office",
            "book_number",
            "volume_number",
            "page_number",
            "serial_number",
            "document_type",
        ],
        "financial": [
            "stamp_duty",
            "registration_fee",
            "market_value"
        ]
    }

    # Fields that LLM understands better
    LLM_PRIORITY = {
        "property": [
            "district",
            "village",
            "survey_number",
            "plot_number",
            "khasra_number",
            "khata_number",
            "area",
            "boundary",
            "sub_district",
            "taluka",
            "tehsil"
        ]
    }

    def merge(self, llm_data, regex_data):

        merged = deepcopy(llm_data)

        # Merge document details
        self._merge_section(
            merged.get("document_details", {}),
            regex_data.get("document_details", {}),
            self.REGEX_PRIORITY["document_details"]
        )

        # Merge financial
        self._merge_section(
            merged.get("financial", {}),
            regex_data.get("financial", {}),
            self.REGEX_PRIORITY["financial"]
        )

        # Merge property
        self._merge_property(
            merged.get("property", {}),
            regex_data.get("property", {})
        )

        # Merge party roles
        self._merge_lists(
            merged.get("party_roles", {}),
            regex_data.get("party_roles", {})
        )

        return merged

    def _merge_section(self, target, source, priority_fields):

        for field in priority_fields:

            regex_value = source.get(field)

            if regex_value:
                target[field] = regex_value

    def _merge_property(self, target, source):

        for field, value in source.items():

            if not value:
                continue

            if not target.get(field):
                target[field] = value

    def _merge_lists(self, target, source):

        for field, values in source.items():

            if values and not target.get(field):
                target[field] = values
