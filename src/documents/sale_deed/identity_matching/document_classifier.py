import re


class DocumentClassifier:
    def __init__(self):
        self.document_patterns = {
            "Sale Deed": [
                r"\bSALE DEED\b",
                r"\bVENDOR\b",
                r"\bPURCHASER\b",
                r"\bSTAMP DUTY\b",
                r"\bREGISTRATION\b",
                r"\bCONSIDERATION\b"
            ],

            "Aadhaar": [
                r"\bUNIQUE IDENTIFICATION AUTHORITY OF INDIA\b",
                r"\bAADHAAR\b",
                r"\bDOB\b",
                r"\bYEAR OF BIRTH\b"
            ],

            "PAN": [
                r"\bINCOME TAX DEPARTMENT\b",
                r"\bPERMANENT ACCOUNT NUMBER\b",
                r"[A-Z]{5}[0-9]{4}[A-Z]"
            ]
        }

    def classify(self, text: str):

        text = text.upper()

        scores = {}

        for document, patterns in self.document_patterns.items():

            score = 0

            for pattern in patterns:
                if re.search(pattern, text):
                    score += 1

            scores[document] = score

        best_document = max(scores, key=scores.get)

        confidence = int(
            (scores[best_document] /
             len(self.document_patterns[best_document])) * 100
        )

        if scores[best_document] == 0:
            return {
                "document_type": "Unknown",
                "confidence": 0
            }

        return {
            "document_type": best_document,
            "confidence": confidence
        }
