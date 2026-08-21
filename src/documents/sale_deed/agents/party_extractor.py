"""
party_extractor.py â€” PartyMapper for SaleDeedPipeline
Wraps the new PartyExtractor with the legacy interface.
"""

import json
from typing import Dict, List


class PartyMapper:
    """
    Maps seller/buyer into all party roles.
    Compatible with the existing SaleDeedPipeline.
    """

    def __init__(self):
        pass

    def extract_json(self, ocr_text: str) -> str:
        """
        Legacy method. Returns JSON string of party extraction.
        Stub for pipeline compatibility.
        """
        return json.dumps({
            "executed_by": [],
            "presented_by": [],
            "vendor": [],
            "seller": [],
            "buyer": [],
            "purchaser": [],
            "vendee": [],
            "transferor": [],
            "transferee": [],
            "executant": [],
            "presenter": [],
            "identifier": [],
            "witness": [],
            "claimant": []
        }, ensure_ascii=False, indent=2)

    def map_roles(self, merged_data: dict) -> dict:
        """
        Map seller/buyer into all party roles.
        Compatible with the existing pipeline.
        """

        if not isinstance(merged_data, dict):
            return merged_data

        seller = merged_data.get("seller", [])
        buyer = merged_data.get("buyer", [])

        def names(lst):
            result = []
            for item in lst:
                if isinstance(item, dict):
                    n = item.get("name", "").strip()
                else:
                    n = str(item).strip()

                if n:
                    result.append(n)

            return result

        sellers = names(seller)
        buyers = names(buyer)

        merged_data["party_roles"] = {
            "executed_by": sellers,
            "vendor": sellers,
            "seller": sellers,
            "transferor": sellers,
            "executant": sellers,

            "buyer": buyers,
            "purchaser": buyers,
            "vendee": buyers,
            "transferee": buyers,

            "presented_by": [],
            "presenter": [],
            "identifier": [],
            "witness": [],
            "claimant": []
        }

        return merged_data
