"""
==============================================================
ITR Structure Detector - Test Suite
==============================================================

Tests StructureDetector independently against the real ITR.

Author : SBFC Document Intelligence
==============================================================
"""

from pathlib import Path

import fitz

from src.documents.itr.detection.structure_detector import (
    StructureDetector,
)


# ==========================================================
# PATH
# ==========================================================

PDF_PATH = (
    Path(__file__).resolve().parent
    / "Vedant ITR.pdf"
)


# ==========================================================
# TEXT EXTRACTION
# ==========================================================

def extract_text(
    pdf_path: Path,
) -> str:
    """
    Extract native PDF text.

    Detection currently uses native text extraction.
    OCR will be handled separately later.
    """

    document = fitz.open(
        str(pdf_path)
    )

    try:

        text_parts = []

        # Analyze first 3 pages for consistency
        max_pages = min(
            3,
            len(document),
        )

        for page_number in range(
            max_pages
        ):

            page = document.load_page(
                page_number
            )

            page_text = page.get_text(
                "text"
            )

            if page_text:

                text_parts.append(
                    page_text
                )

        return "\n".join(
            text_parts
        )

    finally:

        document.close()


# ==========================================================
# TEST
# ==========================================================

def main() -> None:

    print()
    print("#" * 100)
    print("ITR STRUCTURE DETECTOR TEST")
    print("#" * 100)

    print()
    print("PDF:")
    print(PDF_PATH)

    # ------------------------------------------------------
    # File check
    # ------------------------------------------------------

    if not PDF_PATH.exists():

        print()
        print("ERROR: PDF NOT FOUND")
        print()

        return

    # ------------------------------------------------------
    # Extract text
    # ------------------------------------------------------

    text = extract_text(
        PDF_PATH
    )

    print()
    print(
        "Extracted Text Length:",
        len(text),
    )

    # ------------------------------------------------------
    # Detector
    # ------------------------------------------------------

    detector = StructureDetector()

    result = detector.analyze(
        text
    )

    # ======================================================
    # RESULT
    # ======================================================

    print()
    print("=" * 100)
    print("STRUCTURE RESULT")
    print("=" * 100)

    print()
    print(
        "Score:",
        result.score,
    )

    # ------------------------------------------------------
    # Components
    # ------------------------------------------------------

    print()
    print("Detected Components:")
    print("-" * 100)

    if result.detected_components:

        for component in (
            result.detected_components
        ):

            print(
                "-",
                component,
            )

    else:

        print(
            "- None"
        )

    # ------------------------------------------------------
    # Relationships
    # ------------------------------------------------------

    print()
    print("Relationships Found:")
    print("-" * 100)

    if result.relationships_found:

        for relationship in (
            result.relationships_found
        ):

            print(
                "-",
                relationship,
            )

    else:

        print(
            "- None"
        )

    # ------------------------------------------------------
    # Reasons
    # ------------------------------------------------------

    print()
    print("Reasons:")
    print("-" * 100)

    for reason in result.reasons:

        print(
            "-",
            reason,
        )

    # ------------------------------------------------------
    # Full model
    # ------------------------------------------------------

    print()
    print("Full Result:")
    print("-" * 100)

    print(
        result.model_dump()
    )

    print()
    print("#" * 100)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()