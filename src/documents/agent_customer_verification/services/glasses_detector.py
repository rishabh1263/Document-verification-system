"""
Glasses Detection Service.

Uses Roboflow inference to detect spectacles/glasses.
The Roboflow client is initialized lazily so this module
does not break the complete Document Verification API
during application startup.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient


# ============================================================
# PROJECT ROOT
# ============================================================

# Current file:
#
# document-verification-system/
#   src/
#     documents/
#       agent_customer_verification/
#         services/
#           glasses_detector.py
#
# parents[4] = document-verification-system
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

ENV_FILE = PROJECT_ROOT / ".env"


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(
    dotenv_path=ENV_FILE,
    override=False,
)


# ============================================================
# CONFIG
# ============================================================

API_KEY = os.getenv("ROBOFLOW_API_KEY")

MODEL_ID = os.getenv(
    "ROBOFLOW_MODEL_ID",
    "detect-eye-glasses/3",
)

CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# LAZY ROBOFLOW CLIENT
# ============================================================

_client = None


def _get_client():
    """
    Create Roboflow client only when glasses detection
    is actually requested.

    This prevents a missing Roboflow key from crashing
    the entire FastAPI application during import.
    """

    global _client

    if _client is not None:
        return _client

    if not API_KEY:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not configured. "
            f"Expected environment configuration from: {ENV_FILE}"
        )

    _client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY,
    )

    return _client


# ============================================================
# GLASSES DETECTION
# ============================================================

def detect_glasses(image_path: str):
    """
    Detect glasses in an image.

    Returns a stable dictionary even if Roboflow fails.
    """

    try:

        if not image_path:
            return {
                "glasses": False,
                "confidence": 0.0,
                "predictions": [],
                "error": "Image path is missing.",
            }

        path = Path(image_path)

        if not path.is_file():
            return {
                "glasses": False,
                "confidence": 0.0,
                "predictions": [],
                "error": f"Image not found: {image_path}",
            }

        client = _get_client()

        result = client.infer(
            str(path),
            model_id=MODEL_ID,
        )

        predictions = result.get(
            "predictions",
            [],
        )

        if not predictions:
            return {
                "glasses": False,
                "confidence": 0.0,
                "predictions": [],
                "error": None,
            }

        valid_predictions = [
            prediction
            for prediction in predictions
            if float(
                prediction.get(
                    "confidence",
                    0,
                )
            ) >= CONFIDENCE_THRESHOLD
        ]

        if not valid_predictions:
            return {
                "glasses": False,
                "confidence": 0.0,
                "predictions": predictions,
                "error": None,
            }

        best_prediction = max(
            valid_predictions,
            key=lambda prediction: float(
                prediction.get(
                    "confidence",
                    0,
                )
            ),
        )

        return {
            "glasses": True,

            "confidence": round(
                float(
                    best_prediction.get(
                        "confidence",
                        0,
                    )
                ),
                4,
            ),

            "predictions": valid_predictions,

            "error": None,
        }

    except Exception as exc:

        return {
            "glasses": False,
            "confidence": 0.0,
            "predictions": [],
            "error": str(exc),
        }