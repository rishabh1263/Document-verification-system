from dataclasses import asdict
import os
import shutil
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from src.documents.sale_deed.identity_matching.comparator import IdentityComparator
from src.documents.sale_deed.schemas.identity_matching import IdentityComparisonRequest

from src.documents.sale_deed.pipeline import SaleDeedPipeline
from src.documents.sale_deed.aadhaar_pipeline import AadhaarPipeline

router = APIRouter(
    prefix="/identity",
    tags=["Identity Matching"]
)

engine = IdentityComparator()

sale_deed_pipeline = SaleDeedPipeline()
aadhaar_pipeline = AadhaarPipeline()


# ==========================================================
# Existing JSON Comparison API
# ==========================================================

@router.post("/compare")
def compare_identity(request: IdentityComparisonRequest):

    try:

        report = engine.compare(

            source_document=request.source_document,

            target_document=request.target_document,

            source_document_type=request.source_document_type,

            target_document_type=request.target_document_type,

            role=request.role

        )

        return {

            "success": True,

            "message": "Comparison completed successfully",

            "data": asdict(report)

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# NEW FILE UPLOAD API
# ==========================================================

@router.post("/verify")
async def verify_identity(

    sale_deed: UploadFile = File(...),

    aadhaar: UploadFile = File(...),

    role: str = Form("buyer")

):

    sale_deed_path = None
    aadhaar_path = None

    try:

        temp_dir = tempfile.mkdtemp()

        sale_deed_path = os.path.join(
            temp_dir,
            sale_deed.filename
        )

        aadhaar_path = os.path.join(
            temp_dir,
            aadhaar.filename
        )

        with open(sale_deed_path, "wb") as buffer:
            shutil.copyfileobj(sale_deed.file, buffer)

        with open(aadhaar_path, "wb") as buffer:
            shutil.copyfileobj(aadhaar.file, buffer)

        # --------------------------------------------
        # Run Sale Deed Pipeline
        # --------------------------------------------

        sale_result = sale_deed_pipeline.run(
            sale_deed_path
        )

        if not sale_result["success"]:

            raise HTTPException(
                status_code=400,
                detail="Sale Deed processing failed."
            )

        # --------------------------------------------
        # Run Aadhaar Pipeline
        # --------------------------------------------

        aadhaar_result = aadhaar_pipeline.run(
            aadhaar_path
        )

        if not aadhaar_result["success"]:

            raise HTTPException(
                status_code=400,
                detail="Aadhaar processing failed."
            )

        # --------------------------------------------
        # Identity Comparison
        # --------------------------------------------

        report = engine.compare(

            source_document=sale_result["extraction"],

            target_document=aadhaar_result["extraction"],

            source_document_type="sale_deed",

            target_document_type="aadhaar",

            role=role

        )

        return {

            "success": True,

            "message": "Identity verification completed successfully.",

            "sale_deed": sale_result,

            "aadhaar": aadhaar_result,

            "comparison": asdict(report)

        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if sale_deed.file:
            sale_deed.file.close()

        if aadhaar.file:
            aadhaar.file.close()

        if sale_deed_path and os.path.exists(sale_deed_path):
            os.remove(sale_deed_path)

        if aadhaar_path and os.path.exists(aadhaar_path):
            os.remove(aadhaar_path)

        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            os.rmdir(temp_dir)
