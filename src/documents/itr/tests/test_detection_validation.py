"""
ITR Detection -> Validation Integration Test

Runs the complete ITR workflow against the current test documents.

Flow:
    Detection
        |
        +-- NOT ITR --> SKIP validation
        |
        +-- ITR -----> ITR Validation

Expected:
    Vedant ITR.pdf            -> ITR -> VALID
    Canara Bank Statement.pdf -> NOT ITR -> SKIPPED
    KOTAK BANK STATEMENT.pdf  -> NOT ITR -> SKIPPED
    SBI Bank Statement.pdf    -> NOT ITR -> SKIPPED
    Vaibhav salary slip.pdf   -> NOT ITR -> SKIPPED
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from src.documents.itr.detection.detector import ITRDetector
from src.documents.itr.models import DocumentType
from src.documents.itr.validation.models import ValidationDecision
from src.documents.itr.validation.validation_engine import ValidationEngine


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SAMPLES_DIR = PROJECT_ROOT / "samples"
ITR_TEST_DIR = Path(__file__).resolve().parent


TEST_CASES = [
    (
        ITR_TEST_DIR / "Vedant ITR.pdf",
        True,
        "ITR",
    ),
    (
        SAMPLES_DIR / "Canara Bank Statement.pdf",
        False,
        "NOT ITR",
    ),
    (
        SAMPLES_DIR / "KOTAK BANK STATEMENT.pdf",
        False,
        "NOT ITR",
    ),
    (
        SAMPLES_DIR / "SBI Bank Statement.pdf",
        False,
        "NOT ITR",
    ),
    (
        SAMPLES_DIR / "Vaibhav salary slip.pdf",
        False,
        "NOT ITR",
    ),
]


def print_separator(char: str = "=", width: int = 100) -> None:
    print(char * width)


def main() -> None:
    detector = ITRDetector()
    validator = ValidationEngine()

    total = 0
    passed = 0

    print_separator()
    print("ITR DETECTION -> VALIDATION INTEGRATION TEST")
    print_separator()
    print()

    for file_path, expected_itr, expected_label in TEST_CASES:
        total += 1

        print_separator("-")
        print(f"File     : {file_path}")
        print(f"Expected : {expected_label}")
        print()

        if not file_path.exists():
            print("ERROR    : File not found")
            print("Test     : FAIL")
            print()
            continue

        start = perf_counter()

        try:
            detection = detector.detect(str(file_path))

            detection_time = round(
                (perf_counter() - start) * 1000,
                2,
            )

            detected_itr = (
                detection.detected
                and detection.document_type == DocumentType.ITR
            )

            detection_pass = (
                detected_itr == expected_itr
            )

            print("## Detection")
            print(
                f"Detected       : {detection.detected}"
            )
            print(
                f"Document Type  : {detection.document_type}"
            )
            print(
                f"Mode           : {detection.mode}"
            )
            print(
                f"Confidence     : {detection.confidence}"
            )
            print(
                f"Processing ms  : {detection_time}"
            )

            # --------------------------------------------------
            # Non-ITR documents must stop here.
            # --------------------------------------------------

            if not expected_itr:
                validation_skipped = not detected_itr
                test_pass = (
                    detection_pass
                    and validation_skipped
                )

                print()
                print("## Validation")
                print(
                    "Status         : SKIPPED "
                    "(document is not an ITR)"
                )

                print()
                print(
                    f"Test Result    : "
                    f"{'PASS' if test_pass else 'FAIL'}"
                )

                if test_pass:
                    passed += 1

                continue

            # --------------------------------------------------
            # ITR document -> run validation.
            # --------------------------------------------------

            validation = validator.validate_file(
                str(file_path),
                detection_confidence=detection.confidence,
            )

            print()
            print("## Validation")
            print(
                f"Status         : {validation.status}"
            )
            print(
                f"Decision       : {validation.decision}"
            )
            print(
                f"Valid          : {validation.valid}"
            )
            print(
                f"Confidence     : {validation.confidence}"
            )
            print(
                "Integrity      : "
                f"{validation.evidence.integrity.score}"
            )
            print(
                "Content        : "
                f"{validation.evidence.content.score}"
            )
            print(
                "Consistency    : "
                f"{validation.evidence.consistency.score}"
            )
            print(
                f"Processing ms  : "
                f"{validation.processing_time_ms}"
            )

            validation_pass = (
                validation.valid
                and validation.decision
                == ValidationDecision.VALID
            )

            test_pass = (
                detection_pass
                and validation_pass
            )

            print()
            print(
                f"Test Result    : "
                f"{'PASS' if test_pass else 'FAIL'}"
            )

            if test_pass:
                passed += 1

        except Exception as exc:
            print()
            print("ERROR:")
            print(exc)
            print()
            print("Test Result    : FAIL")

    print()
    print_separator()
    print("INTEGRATION TEST SUMMARY")
    print_separator()
    print(f"Total    : {total}")
    print(f"Passed   : {passed}")
    print(f"Failed   : {total - passed}")

    accuracy = (
        (passed / total) * 100
        if total
        else 0.0
    )

    print(f"Accuracy : {accuracy:.2f}%")
    print_separator()

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()