from src.documents.sale_deed.agents.extraction_agent import ExtractionAgent
from .base_extractor import BaseExtractor


class SaleDeedExtractor(BaseExtractor):

    def __init__(self):
        self.agent = ExtractionAgent()

    def extract(self, text: str):
        result = self.agent.extract(text)

        if result.get("success"):
            data = result["data"]

            # Flag empty property extraction for downstream review
            prop = data.get("property", {})
            if all(v == "" for v in prop.values()):
                data["_meta"] = {"property_extraction_failed": True}

            # Flag missing financials
            fin = data.get("financial", {})
            missing = [k for k, v in fin.items() if v == ""]
            if missing:
                data.setdefault("_meta", {})
                data["_meta"]["missing_financial_fields"] = missing

            return data

        return {
            "document_type": "Sale Deed",
            "error": result.get("errors", "Unknown extraction error")
        }
