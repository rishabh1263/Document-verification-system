from pathlib import Path
import shutil
import uuid

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from src.documents.agent_property_verification.services.verification_service import (
    verification_service,
)


# ==========================================
# ROUTER
# ==========================================

router = APIRouter(
    prefix="/verify",
    tags=["Verification"],
)


# ==========================================
# UPLOAD DIRECTORY
# ==========================================

UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================
# VERIFY ENDPOINT
# ==========================================

@router.post("/")
async def verify(
    image: UploadFile = File(...)
):
    """
    Verify agent and property from one
    uploaded camera selfie.

    DEBUG MODE:
    Uploaded image is intentionally preserved
    inside /uploads so we can inspect the exact
    frame received from the React camera.
    """

    image_path = None

    try:

        # ======================================
        # 1. VALIDATE IMAGE
        # ======================================

        if not image.filename:

            raise HTTPException(
                status_code=400,
                detail="Image filename is missing."
            )


        # ======================================
        # 2. VALIDATE CONTENT TYPE
        # ======================================

        allowed_types = {
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/webp",
        }

        if (
            image.content_type
            and
            image.content_type
            not in allowed_types
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported image format: "
                    f"{image.content_type}"
                )
            )


        # ======================================
        # 3. GET EXTENSION
        # ======================================

        extension = (
            Path(image.filename)
            .suffix
            .lower()
        )

        if extension not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }:

            extension = ".jpg"


        # ======================================
        # 4. GENERATE UNIQUE FILE
        # ======================================

        filename = (
            f"{uuid.uuid4()}{extension}"
        )

        image_path = (
            UPLOAD_DIR
            /
            filename
        )


        # ======================================
        # 5. SAVE CAMERA IMAGE
        # ======================================

        with image_path.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                image.file,
                buffer,
            )


        # ======================================
        # DEBUG
        # ======================================

        print()
        print("=" * 60)
        print("CAMERA IMAGE RECEIVED")
        print("=" * 60)

        print(
            f"Filename: {filename}"
        )

        print(
            f"Saved at: {image_path.resolve()}"
        )

        print(
            f"Content type: {image.content_type}"
        )

        print("=" * 60)
        print()


        # ======================================
        # 6. VERIFY IMAGE
        # ======================================

        result = (
            verification_service.verify(
                str(image_path)
            )
        )


        # ======================================
        # 7. DEBUG RESULT
        # ======================================

        print()
        print("=" * 60)
        print("VERIFICATION COMPLETED")
        print("=" * 60)

        print(
            f"Image preserved: "
            f"{image_path.resolve()}"
        )

        print(
            "Verification passed:",
            result
            .get(
                "verification",
                {}
            )
            .get(
                "passed",
                False
            )
        )

        print("=" * 60)
        print()


        # ======================================
        # 8. RETURN RESULT
        # ======================================

        return result


    # ==========================================
    # FASTAPI ERRORS
    # ==========================================

    except HTTPException:

        raise


    # ==========================================
    # INTERNAL ERRORS
    # ==========================================

    except Exception as e:

        print()
        print("=" * 60)
        print("VERIFICATION ERROR")
        print("=" * 60)
        print(str(e))
        print("=" * 60)
        print()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    # ==========================================
    # CLEAN UP UPLOAD HANDLE
    # ==========================================

    finally:

        await image.close()

        # ======================================
        # DEBUG MODE
        # ======================================
        #
        # DO NOT DELETE image_path here.
        #
        # We need the exact camera image for
        # debugging property detection.
        #
        # Once debugging is finished, restore:
        #
        # if (
        #     image_path is not None
        #     and image_path.exists()
        # ):
        #     image_path.unlink(
        #         missing_ok=True
        #     )
        #
        # ======================================
