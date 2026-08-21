from src.documents.sale_deed.identity_matching.extractors.sale_deed_extractor import SaleDeedExtractor
from src.documents.sale_deed.identity_matching.extractors.aadhaar_extractor import AadhaarExtractor
from src.documents.sale_deed.identity_matching.extractors.pan_extractor import PANExtractor


class FieldExtractor:

    def __init__(self):

        self.extractors = {
            "Sale Deed": SaleDeedExtractor(),
            "Aadhaar": AadhaarExtractor(),
            "PAN": PANExtractor()
        }

    def extract(self, document_type, text):

        extractor = self.extractors.get(document_type)

        if extractor is None:
            return {
                "document_type": "Unknown"
            }

        return extractor.extract(text)
