import re


class KeyInformationExtractor:
    """
    Extracts important OCR sections before sending to the LLM.
    PRESERVES full financial and property blocks instead of line-by-line filtering.
    """

    IMPORTANT_KEYWORDS = [
        "sale deed", "gift deed", "lease deed", "mortgage deed",
        "conveyance deed", "partition deed", "relinquishment deed",
        "vendor", "seller", "buyer", "purchaser", "vendee",
        "executant", "executed by", "presented by", "in favour of",
        "identifier", "witness", "transferor", "transferee",
        "registration", "registration no", "registration number",
        "registration office", "sub registrar", "registrar",
        "document no", "deed no", "token no", "serial no",
        "book no", "book number", "volume", "page",
        "property", "property address", "survey", "survey no",
        "survey number", "plot", "plot no", "khasra", "khata",
        "gat", "village", "district", "taluka", "tehsil",
        "boundary", "north", "south", "east", "west",
        "stamp duty", "registration fee", "sale consideration",
        "consideration", "market value", "bank challan",
        "date", "execution", "à¤®à¥‚à¤²à¥à¤¯", "à¤¶à¥à¤²à¥à¤•", "à¤°à¥à¤ªà¤¯à¥‡",
        "à¤°à¤œà¤¿à¤¸à¥à¤Ÿà¥à¤°à¥‡à¤¶à¤¨", "à¤ªà¤‚à¤œà¥€à¤•à¤°à¤£", "à¤¬à¤¿à¤•à¥à¤°à¥€", "à¤µà¤¿à¤•à¥à¤°à¤¯"
    ]

    # Headers that start a block we want to keep entirely
    BLOCK_HEADERS = [
        r"stamp\s+duty",
        r"registration\s+fee",
        r"sale\s+consideration",
        r"consideration",
        r"market\s+value",
        r"financial\s+details",
        r"payment\s+details",
        r"schedule\s+of\s+property",
        r"property\s+details",
        r"land\s+details",
        r"description\s+of\s+property",
        r"à¤µà¤¿à¤µà¤°à¤£",
        r"à¤œà¤®à¥€à¤¨",
        r"à¤®à¥‚à¤²à¥à¤¯",
        r"à¤¶à¥à¤²à¥à¤•",
        r"à¤°à¥à¤ªà¤¯à¥‡",
    ]

    PARTY_PATTERN = re.compile(
        r"(executed by|executant|presented by|in favour of|identifier|witness)",
        re.IGNORECASE
    )

    BLOCK_PATTERN = re.compile(
        r"|".join(BLOCK_HEADERS),
        re.IGNORECASE
    )

    PAGE_PATTERN = re.compile(
        r"========== PAGE \d+ ==========",
        re.IGNORECASE
    )

    def clean(self, line):
        line = re.sub(r"\s+", " ", line)
        return line.strip()

    def is_noise(self, line):
        line = line.strip()
        if not line:
            return True
        if "Scanned with" in line:
            return True
        if len(line) <= 2:
            return True
        letters = sum(c.isalpha() for c in line)
        digits = sum(c.isdigit() for c in line)
        if letters == 0 and digits < 3:
            return True
        if letters < 2 and len(line) < 10:
            return True
        return False

    def is_important(self, line):
        line = line.lower()
        return any(keyword in line for keyword in self.IMPORTANT_KEYWORDS)

    def is_block_header(self, line):
        """Check if line starts a financial/property block."""
        return bool(self.BLOCK_PATTERN.search(line))

    def extract(self, ocr_text):
        if not ocr_text:
            return ""

        lines = []
        for raw in ocr_text.splitlines():
            line = self.clean(raw)
            if self.is_noise(line):
                continue
            lines.append(line)

        selected = []
        total = len(lines)
        i = 0

        while i < total:
            line = lines[i]
            lower = line.lower()

            # Keep page headers
            if self.PAGE_PATTERN.match(line):
                selected.append(line)
                i += 1
                continue

            # Keep complete party blocks
            if self.PARTY_PATTERN.search(lower):
                j = i
                while j < total:
                    current = lines[j]
                    current_lower = current.lower()
                    if (
                        j != i
                        and (
                            self.PAGE_PATTERN.match(current)
                            or self.PARTY_PATTERN.search(current_lower)
                        )
                    ):
                        break
                    selected.append(current)
                    j += 1
                i = j
                continue

            # Keep complete financial/property blocks (MUCH larger window)
            if self.is_block_header(line):
                start = max(0, i - 2)
                end = min(total, i + 50)  # â† Was 6, now 50 lines!
                for k in range(start, end):
                    selected.append(lines[k])
                i = end
                continue

            # Keep important keyword lines (smaller window for general keywords)
            if self.is_important(line):
                start = max(0, i - 1)
                end = min(total, i + 6)
                for k in range(start, end):
                    selected.append(lines[k])

            i += 1

        # Remove duplicates while preserving order
        unique = []
        seen = set()
        for line in selected:
            key = line.lower()
            if key not in seen:
                unique.append(line)
                seen.add(key)

        result = "\n".join(unique)

        print("\n" + "=" * 70)
        print("KEY INFORMATION EXTRACTION")
        print("=" * 70)
        print(f"Original Characters : {len(ocr_text)}")
        print(f"Filtered Characters : {len(result)}")
        if len(ocr_text):
            reduction = ((len(ocr_text) - len(result)) / len(ocr_text)) * 100
            print(f"Reduction %         : {reduction:.2f}%")
        print("=" * 70)

        return result
