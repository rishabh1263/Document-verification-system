"""
ExtractionAgent â€” Adapter for SaleDeedPipeline
Wraps ExtractionEngine and converts output to the format pipeline.py expects.
"""

from src.documents.sale_deed.extraction_engine import ExtractionEngine


class ExtractionAgent:
    """
    Drop-in replacement for the old ExtractionAgent.
    Runs the new modular engine and flattens rich field objects
    into plain strings/arrays that pipeline.py expects.
    """

    def __init__(self):
        self.engine = ExtractionEngine()

    def extract(self, ocr_text: str) -> dict:
        """
        Main entry point called by pipeline.py.
        Returns {"success": True, "data": {...}} to match old interface.
        """
        engine_result = self.engine.run(ocr_text)
        adapted = self._adapt_to_pipeline(engine_result)
        return {"success": True, "data": adapted}

    def _adapt_to_pipeline(self, engine_result: dict) -> dict:
        parties = engine_result.get("parties", {})

        # Resolve seller / buyer lists
        sellers = (
            parties.get("seller", [])
            or parties.get("vendor", [])
            or parties.get("executed_by", [])
            or parties.get("executant", [])
        )
        buyers = (
            parties.get("buyer", [])
            or parties.get("purchaser", [])
            or parties.get("vendee", [])
            or parties.get("transferee", [])
        )

        # Deduplicate while preserving order
        sellers = list(dict.fromkeys(sellers))
        buyers = list(dict.fromkeys(buyers))

        # Build party_roles exactly as PartyMapper used to
        party_roles = {
            "executed_by": sellers,
            "vendor": sellers,
            "seller": sellers,
            "transferor": sellers,
            "executant": sellers,
            "buyer": buyers,
            "purchaser": buyers,
            "vendee": buyers,
            "transferee": buyers,
            "presented_by": parties.get("presented_by", []),
            "presenter": parties.get("presenter", []),
            "identifier": parties.get("identifier", []),
            "witness": parties.get("witness", []),
            "claimant": parties.get("claimant", [])
        }

        return {
            "document_details": self._flatten_section(
                engine_result.get("document_details", {})
            ),
            "financial": self._flatten_section(
                engine_result.get("financial", {})
            ),
            "property": self._flatten_section(
                engine_result.get("property", {})
            ),
            "seller": sellers,
            "buyer": buyers,
            "party_roles": party_roles,
            "_overall_confidence": engine_result.get("overall_confidence", 0),
            "_metadata": engine_result.get("metadata", {})
        }

    def _flatten_section(self, section: dict) -> dict:
        flat = {}
        for k, v in section.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict) and "value" in v:
                flat[k] = v["value"]
            elif isinstance(v, list):
                flat[k] = [i["value"] if isinstance(i, dict) and "value" in i else i for i in v]
            else:
                flat[k] = v
        return flat
