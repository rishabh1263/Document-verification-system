import re


class PageClassifier:
    """
    Splits OCR into pages and keeps only useful ones.
    """

    PROPERTY_KEYWORDS = [

        "survey",
        "plot",
        "khasra",
        "khata",
        "boundary",
        "north",
        "south",
        "east",
        "west",
        "village",
        "district"

    ]

    PAGE_REGEX = r"========== PAGE (\d+) =========="

    def split_pages(self, text):

        matches = list(re.finditer(self.PAGE_REGEX, text))

        pages = {}

        for i, match in enumerate(matches):

            page_no = int(match.group(1))

            start = match.end()

            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            pages[page_no] = text[start:end].strip()

        return pages

    def keep_page(self, page_no, text):

        # Always keep registration summary

        if page_no == 1:

            return True

        # Always keep party page

        if page_no == 2:

            return True

        # Always keep endorsements

        if page_no == 7:

            return True

        lower = text.lower()

        for keyword in self.PROPERTY_KEYWORDS:

            if keyword in lower:

                return True

        return False

    def filter(self, text):

        pages = self.split_pages(text)

        result = []

        for page_no, page_text in pages.items():

            if self.keep_page(page_no, page_text):

                result.append(f"========== PAGE {page_no} ==========")

                result.append(page_text)

        return "\n".join(result)
