import fitz
from pathlib import Path


class PDFTextExtractor:

    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)

    def extract_text(self):

        if not self.pdf_path.exists():
            raise FileNotFoundError(self.pdf_path)

        doc = fitz.open(self.pdf_path)

        text = ""

        for page in doc:
            text += page.get_text()

        doc.close()

        return text.strip()

    def has_embedded_text(self, min_chars=100):

        text = self.extract_text()

        return len(text) >= min_chars, text
