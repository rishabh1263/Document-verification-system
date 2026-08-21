from pathlib import Path

from src.common.face.detector import FaceDetector
from src.common.pdf.renderer import render_pdf_pages


def main():
    project_root = Path(__file__).resolve().parents[2]

    pdf_path = (
        project_root
        / "samples"
        / "driving_license.pdf"
    )

    print("\n--- Driving Licence Phase 1 ---")
    print(f"PDF: {pdf_path}")

    with open(pdf_path, "rb") as file:
        pdf_bytes = file.read()

    pages = render_pdf_pages(
        file_bytes=pdf_bytes,
        dpi=150,
    )

    print(
        f"Pages rendered: {len(pages)}"
    )

    if not pages:
        raise RuntimeError(
            "No pages found."
        )

    page_image = pages[0]

    # Load model only once.
    detector = FaceDetector()

    print("\n--- Detection Benchmark ---")

    for run_number in range(1, 6):

        result = detector.detect_image(
            page_image
        )

        print(
            f"\nRun {run_number}:"
        )

        print(
            f"  Face detected: "
            f"{result.detected}"
        )

        print(
            f"  Face count: "
            f"{result.face_count}"
        )

        print(
            f"  Processing time: "
            f"{result.processing_time_ms:.2f} ms"
        )

        for index, face in enumerate(
            result.faces,
            start=1,
        ):
            print(
                f"  Face {index}:"
            )

            print(
                f"    Confidence: "
                f"{face.confidence:.4f}"
            )

            print(
                f"    Bounding box: "
                f"{face.bounding_box}"
            )

            print(
                f"    Area: "
                f"{face.area}"
            )


if __name__ == "__main__":
    main()