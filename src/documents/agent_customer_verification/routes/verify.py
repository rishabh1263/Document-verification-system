from fastapi import APIRouter, UploadFile, File
import os
import shutil
import cv2

from src.documents.agent_customer_verification.services.face_service import FaceService

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------------------------------------------------
# Create services ONCE when FastAPI starts
# --------------------------------------------------

service = FaceService()


@router.post("/verify")
async def verify(
    group_selfie: UploadFile = File(...)
):

    image_path = os.path.join(
        UPLOAD_DIR,
        "group_selfie.jpg"
    )

    try:

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(group_selfie.file, buffer)

        image = cv2.imread(image_path)

        if image is None:
            return {
                "status": "FAILED",
                "message": "Unable to read uploaded image."
            }

        # --------------------------------------------------
        # Step 1 - Image Quality Validation
        # --------------------------------------------------

        validation = service.validate_image(image)

        if not validation["success"]:
            return {
                "status": "FAILED",
                "message": validation["message"],
                "metrics": validation.get("metrics", {})
            }

        # --------------------------------------------------
        # Step 2 - Detect Faces & Extract Embeddings
        # --------------------------------------------------

        face_result = service.embedding_service.get_face_embeddings(
            image_path
        )

        if not face_result["success"]:
            return {
                "status": "FAILED",
                "message": face_result["message"],
                "faces_detected": face_result.get("faces", 0)
            }

        # --------------------------------------------------
        # Step 3 - Verify Agent & Customer
        # --------------------------------------------------

        verification = service.verification_service.verify_group(
            face_result["faces"]
        )

        if not verification["success"]:
            return {
                "status": "FAILED",
                "message": verification["message"],
                "scores": verification.get("scores", [])
            }

        return {
            "status": "SUCCESS",
            "message": "Both Agent and Customer verified successfully.",
            "metrics": validation["metrics"],
            "data": verification
        }

    finally:

        if os.path.exists(image_path):
            os.remove(image_path)
