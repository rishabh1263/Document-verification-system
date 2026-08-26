import cv2

from src.common.face.model import get_face_app


class FaceService:

    def __init__(self):

        print("Loading shared InsightFace Model...")

        self.app = get_face_app()

        print("InsightFace Model Ready")

    def detect_face(self, image_path: str):

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Unable to read image: {image_path}"
            )

        faces = self.app.get(image)

        if len(faces) == 0:
            raise ValueError(
                "No face detected."
            )

        if len(faces) > 1:
            raise ValueError(
                f"Exactly one face is required. Found {len(faces)} faces."
            )

        return {
            "image": image,
            "face": faces[0],
        }


face_service = FaceService()