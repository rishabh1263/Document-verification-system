"""
==============================================================
ITR DETECTION TEST SUITE - V3
==============================================================

Tests the complete ITR Detection pipeline:

    Metadata
    Keyword
    Layout
    Structure
    Confidence

Author : SBFC Document Intelligence
==============================================================
"""

from pathlib import Path

from src.documents.itr.detection.detector import (
    ITRDetector,
)


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = (
    Path(__file__).resolve()
    .parents[4]
)

TEST_DIRECTORY = (
    Path(__file__).resolve().parent
)

SAMPLES_DIRECTORY = (
    PROJECT_ROOT
    / "samples"
)


# ==========================================================
# TEST DOCUMENTS
# ==========================================================

TEST_CASES = [

    {
        "path": TEST_DIRECTORY / "Vedant ITR.pdf",
        "expected": True,
        "label": "ITR",
    },

    {
        "path": SAMPLES_DIRECTORY
        / "Canara Bank Statement.pdf",
        "expected": False,
        "label": "NOT ITR",
    },

    {
        "path": SAMPLES_DIRECTORY
        / "KOTAK BANK STATEMENT.pdf",
        "expected": False,
        "label": "NOT ITR",
    },

    {
        "path": SAMPLES_DIRECTORY
        / "SBI Bank Statement.pdf",
        "expected": False,
        "label": "NOT ITR",
    },

    {
        "path": SAMPLES_DIRECTORY
        / "Vaibhav salary slip.pdf",
        "expected": False,
        "label": "NOT ITR",
    },
]


# ==========================================================
# PRINT RESULT
# ==========================================================

def print_result(
    path: Path,
    expected: bool,
    result,
) -> bool:

    actual = bool(
        result.detected
    )

    passed = (
        actual == expected
    )

    print()
    print("=" * 100)

    print(
        "File     :",
        path,
    )

    print(
        "Expected :",
        "ITR" if expected else "NOT ITR",
    )

    print()
    print("## Actual Result")
    print()

    print(
        "Detected       :",
        result.detected,
    )

    print(
        "Document Type  :",
        result.document_type,
    )

    print(
        "Mode           :",
        result.mode,
    )

    print(
        "Confidence     :",
        result.confidence,
    )

    print(
        "Page Count     :",
        result.page_count,
    )

    print(
        "Processing ms  :",
        round(
            result.processing_time_ms,
            2,
        ),
    )

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    metadata = result.evidence.metadata

    print()
    print("## Metadata Evidence")
    print()

    print(
        "Valid File     :",
        metadata.valid_file,
    )

    print(
        "Supported      :",
        metadata.is_supported,
    )

    print(
        "Valid PDF      :",
        metadata.is_valid_pdf,
    )

    print(
        "Encrypted      :",
        metadata.encrypted,
    )

    print(
        "Corrupted      :",
        metadata.corrupted,
    )

    print(
        "Metadata Score :",
        metadata.score,
    )

    # ------------------------------------------------------
    # Keyword
    # ------------------------------------------------------

    keyword = result.evidence.keyword

    print()
    print("## Keyword Evidence")
    print()

    print(
        "Keyword Score  :",
        keyword.score,
    )

    print(
        "Positive Score :",
        keyword.total_positive_score,
    )

    print(
        "Negative Score :",
        keyword.total_negative_score,
    )

    print(
        "Keywords Found :",
        keyword.total_keywords_found,
    )

    print(
        "Matched        :",
        [
            item.keyword
            for item in keyword.matched_keywords
        ],
    )

    print(
        "Negative       :",
        [
            item.keyword
            for item in keyword.negative_keywords
        ],
    )

    # ------------------------------------------------------
    # Layout
    # ------------------------------------------------------

    layout = result.evidence.layout

    print()
    print("## Layout Evidence")
    print()

    print(
        "Layout Score   :",
        layout.score,
    )

    print(
        "Sections       :",
        layout.detected_sections,
    )

    # ------------------------------------------------------
    # Structure
    # ------------------------------------------------------

    structure = result.evidence.structure

    print()
    print("## Structure Evidence")
    print()

    print(
        "Structure Score:",
        structure.score,
    )

    print(
        "Components     :",
        structure.detected_components,
    )

    print(
        "Relationships  :",
        structure.relationships_found,
    )

    # ------------------------------------------------------
    # Reasons
    # ------------------------------------------------------

    print()
    print("## Reasons")
    print()

    for reason in keyword.reasons:

        print(
            "-",
            reason,
        )

    # ------------------------------------------------------
    # Test result
    # ------------------------------------------------------

    print()
    print("## Test Result")
    print()

    if passed:

        print("PASS")

    else:

        print(
            "FAIL"
        )

        print(
            "Expected detected =",
            expected,
        )

        print(
            "Actual detected   =",
            actual,
        )

    return passed


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:

    print()
    print("#" * 100)
    print("ITR DETECTION TEST SUITE")
    print("#" * 100)

    print()
    print(
        "Project Root:"
    )

    print(
        PROJECT_ROOT
    )

    print()
    print(
        "ITR Test Directory:"
    )

    print(
        TEST_DIRECTORY
    )

    print()
    print(
        "Samples Directory:"
    )

    print(
        SAMPLES_DIRECTORY
    )

    detector = ITRDetector()

    total = 0
    passed = 0
    failed = 0

    # ======================================================
    # RUN TESTS
    # ======================================================

    for case in TEST_CASES:

        total += 1

        path = Path(
            case["path"]
        )

        expected = bool(
            case["expected"]
        )

        # --------------------------------------------------
        # Missing file
        # --------------------------------------------------

        if not path.exists():

            print()
            print("=" * 100)

            print(
                "File     :",
                path,
            )

            print(
                "Expected :",
                case["label"],
            )

            print(
                "STATUS   : FILE NOT FOUND"
            )

            print(
                "TEST     : FAIL"
            )

            failed += 1

            continue

        # --------------------------------------------------
        # Execute
        # --------------------------------------------------

        try:

            result = detector.detect(
                str(path)
            )

            if print_result(
                path=path,
                expected=expected,
                result=result,
            ):

                passed += 1

            else:

                failed += 1

        except Exception as exc:

            failed += 1

            print()
            print("=" * 100)

            print(
                "File     :",
                path,
            )

            print(
                "Expected :",
                case["label"],
            )

            print()
            print(
                "ERROR:"
            )

            print(
                type(exc).__name__,
                ":",
                exc,
            )

    # ======================================================
    # SUMMARY
    # ======================================================

    accuracy = (
        (passed / total) * 100
        if total
        else 0.0
    )

    print()
    print("#" * 100)
    print("TEST SUMMARY")
    print("#" * 100)

    print(
        "Total    :",
        total,
    )

    print(
        "Passed   :",
        passed,
    )

    print(
        "Failed   :",
        failed,
    )

    print(
        "Accuracy :",
        f"{accuracy:.2f}%",
    )

    print("#" * 100)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()