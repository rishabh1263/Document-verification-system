import json
import ollama

from src.documents.sale_deed.agents.prompts import PROMPT_TEMPLATE


class DocumentAnalysisAgent:

    def __init__(self, model="qwen2.5:3b"):
        self.model = model

    def analyze(self, document_text):

        prompt = PROMPT_TEMPLATE.replace(
            "__DOCUMENT_TEXT__",
            document_text
        )

        # ==========================
        # DEBUG: Print Prompt
        # ==========================
        print("\n" + "=" * 80)
        print("PROMPT SENT TO OLLAMA")
        print("=" * 80)
        print(prompt)
        print("=" * 80)

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0,
                "top_p": 0.9,
                "num_predict": 1024,
                "num_ctx": 8192
            }
        )

        answer = response["message"]["content"].strip()

        # ==========================
        # DEBUG: Print Raw Response
        # ==========================
        print("\n" + "=" * 80)
        print("RAW OLLAMA RESPONSE")
        print("=" * 80)
        print(answer)
        print("=" * 80)

        answer = answer.replace("```json", "")
        answer = answer.replace("```", "").strip()

        return json.loads(answer)
