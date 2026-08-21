"""
Document Verification Agent
Uses Ollama + Qwen to determine whether the OCR text
represents an Indian Sale Deed.
"""

import json
import ollama

from src.documents.sale_deed.agents.prompts import DOCUMENT_VERIFICATION_PROMPT

class DocumentVerifier:

    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model

    def verify(self, document_text: str) -> dict:
        """
        Verify whether the OCR text represents a Sale Deed.

        Args:
            document_text: OCR extracted text

        Returns:
            Dictionary containing verification result.
        """

        prompt = DOCUMENT_VERIFICATION_PROMPT.replace(
            "{{OCR_TEXT}}",
            document_text
        )

        try:

            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response["message"]["content"].strip()

            # Remove markdown if returned
            answer = answer.replace("```json", "")
            answer = answer.replace("```", "").strip()

            result = json.loads(answer)

            # Ensure all expected keys exist
            defaults = {
                "verified": False,
                "confidence": 0,
                "document_type": "Unknown",
                "reason": "",
                "evidence": [],
                "missing_elements": [],
                "manual_review_required": False,
            }

            defaults.update(result)

            return defaults

        except json.JSONDecodeError:

            return {
                "verified": False,
                "confidence": 0,
                "document_type": "Unknown",
                "reason": "Model returned invalid JSON.",
                "evidence": [],
                "missing_elements": [],
                "manual_review_required": True,
            }

        except Exception as e:

            return {
                "verified": False,
                "confidence": 0,
                "document_type": "Unknown",
                "reason": str(e),
                "evidence": [],
                "missing_elements": [],
                "manual_review_required": True,
            }
