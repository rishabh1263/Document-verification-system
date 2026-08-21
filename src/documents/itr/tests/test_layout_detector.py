from pathlib import Path

import fitz

from src.documents.itr.detection.layout_detector import (
    LayoutDetector,
)


PDF_PATH = (
    Path(__file__).resolve().parent
    / "Vedant ITR.pdf"
)


def extract_text(pdf_path: Path) -> str:

    document = fitz.open(
        str(pdf_path)
    )

    try:

        text_parts = []

        for page_number in range(
            min(3, len(document))
        ):

            page = document.load_page(
                page_number
            )

            text = page.get_text(
                "text"
            )

            if text:

                text_parts.append(
                    text
                )

        return "\n".join(
            text_parts
        )

    finally:

        document.close()


if __name__ == "__main__":

    print("=" * 100)
    print("ITR LAYOUT DETECTOR TEST")
    print("=" * 100)

    text = extract_text(
        PDF_PATH
    )

    detector = LayoutDetector()

    result = detector.analyze(
        text
    )

    print("\nScore:")
    print(result.score)

    print("\nDetected Sections:")

    for section in result.detected_sections:

        print(
            "-",
            section,
        )

    print("\nReasons:")

    for reason in result.reasons:

        print(
            "-",
            reason,
        )

    print("\nFull Result:")
    print(
        result.model_dump()
    )

    print("=" * 100)