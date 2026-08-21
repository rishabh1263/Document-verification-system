"""
Aadhaar Pipeline

End-to-end pipeline for Aadhaar document processing.
"""

from src.documents.sale_deed.document_loader import DocumentLoader
from src.documents.sale_deed.quality_checker import QualityChecker
from src.documents.sale_deed.image_enhancer import ImageEnhancer
from src.documents.sale_deed.ocr_engine import OCREngine

from src.documents.sale_deed.agents.verification_agent import VerificationAgent
from src.documents.sale_deed.agents.aadhaar_extraction_agent import AadhaarExtractionAgent
from src.documents.sale_deed.agents.aadhaar_validation_agent import AadhaarValidationAgent


class AadhaarPipeline:

    def __init__(self):

        self.document_loader = DocumentLoader()

        self.quality_checker = QualityChecker()

        self.image_enhancer = ImageEnhancer()

        self.ocr = OCREngine()

        self.verification_agent = VerificationAgent()

        self.extraction_agent = AadhaarExtractionAgent()

        self.validation_agent = AadhaarValidationAgent()

    # -----------------------------------------------------------------

    def run(self, document_path: str):

        try:

            # ----------------------------------------------------------
            # Step 1 : Load Document
            # ----------------------------------------------------------

            document = self.document_loader.load(document_path)

            # ----------------------------------------------------------
            # Step 2 : Quality Check
            # ----------------------------------------------------------

            quality = self.quality_checker.check(document)

            if not quality.get("success", True):

                return {

                    "success": False,

                    "stage": "quality_check",

                    "message": "Document quality is too poor.",

                    "details": quality

                }

            # ----------------------------------------------------------
            # Step 3 : Image Enhancement
            # ----------------------------------------------------------

            enhanced_document = self.image_enhancer.enhance(document)

            # ----------------------------------------------------------
            # Step 4 : OCR
            # ----------------------------------------------------------

            ocr_result = self.ocr.extract_text(enhanced_document)

            if isinstance(ocr_result, dict):

                ocr_text = ocr_result.get("text", "")

            else:

                ocr_text = ocr_result

            # ----------------------------------------------------------
            # Step 5 : Verification
            # ----------------------------------------------------------

            verification = self.verification_agent.run(ocr_text)

            if not verification.get("success", True):

                return {

                    "success": False,

                    "stage": "verification",

                    "message": "Document verification failed.",

                    "details": verification

                }

            # ----------------------------------------------------------
            # Step 6 : Extraction
            # ----------------------------------------------------------

            extraction = self.extraction_agent.run(ocr_text)

            if not extraction.get("success", False):

                return {

                    "success": False,

                    "stage": "extraction",

                    "message": extraction.get("message"),

                    "details": extraction

                }

            extracted_data = extraction["data"]

            # ----------------------------------------------------------
            # Step 7 : Validation
            # ----------------------------------------------------------

            validation = self.validation_agent.run(extracted_data)

            # ----------------------------------------------------------
            # Final Response
            # ----------------------------------------------------------

            return {

                "success": validation["success"],

                "document_type": "aadhaar",

                "ocr_text": ocr_text,

                "verification": verification,

                "extraction": extracted_data,

                "validation": validation

            }

        except Exception as e:

            return {

                "success": False,

                "message": str(e)
            }
