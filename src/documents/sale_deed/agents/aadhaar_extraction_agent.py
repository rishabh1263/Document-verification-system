"""
Aadhaar Extraction Agent

This agent acts as a wrapper around AadhaarExtractionEngine.
It is responsible for:
1. Receiving OCR text.
2. Calling the extraction engine.
3. Returning standardized extracted data.
"""
from src.documents.sale_deed.aadhaar.extraction_engine import AadhaarExtractionEngine

class AadhaarExtractionAgent:

    def __init__(self):
        self.engine = AadhaarExtractionEngine()

    def run(self, ocr_text: str) -> dict:
        """
        Run Aadhaar extraction.

        Args:
            ocr_text (str): OCR extracted text.

        Returns:
            dict: Structured Aadhaar extraction output.
        """

        if not ocr_text:
            return {
                "success": False,
                "message": "OCR text is empty.",
                "data": {}
            }

        try:

            result = self.engine.run(ocr_text)

            return {
                "success": True,
                "message": "Aadhaar extraction completed successfully.",
                "data": result
            }

        except Exception as e:

            return {
                "success": False,
                "message": f"Aadhaar extraction failed: {str(e)}",
                "data": {}
            }
