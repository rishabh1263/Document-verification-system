"""
Verification Agent
Determines whether the document is a Sale Deed.
"""

import json
import ollama

from src.documents.sale_deed.agents.prompts import VERIFICATION_PROMPT


class VerificationAgent:

    def __init__(
        self,
        model="qwen2.5:3b",
        max_chars=4000,
        debug=False
    ):
        self.model = model
        self.max_chars = max_chars
        self.debug = debug

    def analyze(self, document_text):

        # Use only the beginning of the OCR for verification
        verification_text = document_text[:self.max_chars]

        prompt = VERIFICATION_PROMPT.replace(
            "__DOCUMENT_TEXT__",
            verification_text
        )

        if self.debug:
            print("=" * 80)
            print("VERIFICATION OCR")
            print("=" * 80)
            print(verification_text)
            print("=" * 80)

        try:

            response = ollama.chat(
                model=self.model,
                keep_alive="10m",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0,
                    "num_predict": 250
                }
            )

            answer = response["message"]["content"].strip()

            if answer.startswith("```json"):
                answer = answer.replace("```json", "").replace("```", "").strip()

            elif answer.startswith("```"):
                answer = answer.replace("```", "").strip()

            if self.debug:
                print("\nLLM RESPONSE")
                print("=" * 80)
                print(answer)
                print("=" * 80)

            return json.loads(answer)

        except Exception as e:

            print(f"Verification Error: {e}")

            return {
                "verified": False,
                "confidence": 0,
                "document_type": "Unknown",
                "reason": "Verification failed",
                "manual_review_required": True
            }
