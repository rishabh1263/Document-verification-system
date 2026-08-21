import re


class NoiseCleaner:
    """
    Cleans raw OCR output before processing.

    Removes:
    - Scanner watermark
    - Empty lines
    - OCR garbage
    - Duplicate spaces
    """

    WATERMARKS = [

        "Scanned with",
        "OKEN Scanner",
        "Computer Operator",
        "Photo & Finger",
        "Thumb",
        "Middle",
        "Ring",
        "Index"

    ]

    def clean_line(self, line):

        line = re.sub(r"\s+", " ", line)

        return line.strip()

    def is_noise(self, line):

        if not line:

            return True

        lower = line.lower()

        for watermark in self.WATERMARKS:

            if watermark.lower() in lower:

                return True

        # very short junk

        if len(line) <= 2:

            return True

        letters = sum(c.isalpha() for c in line)

        digits = sum(c.isdigit() for c in line)

        # symbol-only OCR

        if letters == 0 and digits < 3:

            return True

        # tiny OCR fragments

        if letters < 2 and len(line) < 10:

            return True

        return False

    def clean(self, text):

        cleaned = []

        for raw in text.splitlines():

            line = self.clean_line(raw)

            if self.is_noise(line):

                continue

            cleaned.append(line)

        return "\n".join(cleaned)
