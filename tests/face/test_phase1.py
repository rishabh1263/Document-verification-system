from pathlib import Path

from src.common.face.detector import FaceDetector


def main():
    project_root = Path(__file__).resolve().parents[2]

    image_path = (
        project_root
        / "samples"
        / "view.jpg"
    )

    print(
        "\n--- Phase 1 Face Detection ---"
    )

    print(
        f"Image: {image_path}"
    )

    # Load detector once.
    detector = FaceDetector()

    # Run detection.
    result = detector.detect(
        str(image_path)
    )

    print("\n--- Result ---")

    print(
        f"Face detected: "
        f"{result.detected}"
    )

    print(
        f"Face count: "
        f"{result.face_count}"
    )

    print(
        f"Processing time: "
        f"{result.processing_time_ms:.2f} ms"
    )

    print(
        "\n--- Detected Faces ---"
    )

    if not result.faces:
        print(
            "No faces detected."
        )
        return

    for index, face in enumerate(
        result.faces,
        start=1,
    ):

        print(
            f"Face {index}:"
        )

        print(
            f"  Confidence: "
            f"{face.confidence:.4f}"
        )

        print(
            f"  Bounding box: "
            f"{face.bounding_box}"
        )

        print(
            f"  Area: "
            f"{face.area}"
        )


if __name__ == "__main__":
    main()