import cv2
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from src.common.face.model import get_face_app

from src.documents.agent_customer_verification.exceptions import (
    InvalidImageException,
)
from src.documents.agent_customer_verification.services.glasses_detector import (
    detect_glasses,
)


class EmbeddingService:

    def __init__(self):

        # Use the shared InsightFace buffalo_l instance.
        # This prevents loading the model again.
        self.app = get_face_app()

    def detect_faces(self, image_path: str):

        image = cv2.imread(image_path)

        if image is None:
            raise InvalidImageException()

        faces = self.app.get(image)

        print("\n" + "=" * 60)
        print(f"Image          : {image_path}")
        print(f"Faces Detected : {len(faces)}")
        print("=" * 60)

        return image, faces

    def process_face(self, image, face, index):

        x1, y1, x2, y2 = map(int, face.bbox)

        h, w = image.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = image[y1:y2, x1:x2]

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as tmp:

            temp_path = tmp.name

        cv2.imwrite(temp_path, crop)

        try:
            glasses = detect_glasses(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return {
            "index": index,
            "glasses": glasses,
            "face": {
                "id": index + 1,
                "bbox": face.bbox.tolist(),
                "score": float(face.det_score),
                "embedding": face.embedding,
            },
        }

    def get_face_embeddings(self, image_path: str):

        image, faces = self.detect_faces(image_path)

        if len(faces) != 2:

            return {
                "success": False,
                "message": "Exactly two faces must be present.",
                "faces": len(faces),
            }

        print("\nRunning glasses detection in parallel...\n")

        with ThreadPoolExecutor(max_workers=2) as executor:

            futures = [
                executor.submit(
                    self.process_face,
                    image,
                    face,
                    index,
                )
                for index, face in enumerate(faces)
            ]

            results = [
                future.result()
                for future in futures
            ]

        results.sort(
            key=lambda x: x["index"]
        )

        detected_faces = []

        for result in results:

            glasses = result["glasses"]

            if glasses.get("error"):

                return {
                    "success": False,
                    "message": (
                        f"Glasses detector error: "
                        f"{glasses['error']}"
                    ),
                }

            if glasses["glasses"]:

                return {
                    "success": False,
                    "message": (
                        "Please remove spectacles and "
                        "capture the selfie again."
                    ),
                    "face": result["index"] + 1,
                    "confidence": glasses["confidence"],
                }

            detected_faces.append(
                result["face"]
            )

        print("\n✓ No spectacles detected.")

        return {
            "success": True,
            "image": image,
            "faces": detected_faces,
        }