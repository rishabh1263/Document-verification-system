"""
Error Level Analysis (ELA) forgery signal.

Purpose:
    Detect regions of a document image that may have a different
    compression history from the surrounding image.

Important:
    ELA is a heuristic signal. It does NOT prove that a document
    is genuine or forged.

Improvements over basic ELA:
    1. Global ELA statistics
    2. Percentile-based analysis instead of relying only on max pixel
    3. Patch/local-region analysis
    4. Suspicious-region ratio
    5. Multiple independent signals
    6. Safer scoring to reduce false positives
    7. Unique output files
    8. Structured details for API output
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any, Dict, List

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ELA_QUALITY = 90

# Difference amplification used only for the human-review image.
ELA_AMPLIFICATION = 8

# Patch size for localized analysis.
PATCH_SIZE = 64

# Ignore extremely small images.
MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 200

# Starting thresholds.
#
# These MUST eventually be calibrated against your own genuine and
# tampered salary-slip dataset.
GLOBAL_MEAN_THRESHOLD = 8.0
GLOBAL_STD_THRESHOLD = 12.0

P95_THRESHOLD = 25.0
P99_THRESHOLD = 45.0

PATCH_MEAN_THRESHOLD = 18.0

# Percentage of patches allowed to exceed PATCH_MEAN_THRESHOLD.
SUSPICIOUS_PATCH_RATIO_THRESHOLD = 0.08


# ============================================================
# PUBLIC CHECK
# ============================================================

def check(
    image_path: str,
    output_dir: str | None = None,
) -> Dict[str, Any]:

    result: Dict[str, Any] = {
        "score": 0,
        "reasons": [],
        "checked": False,
        "status": "not_applicable",
        "details": {},
        "ela_image_path": None,
    }

    # ========================================================
    # DEPENDENCIES
    # ========================================================

    try:
        from PIL import Image, ImageChops

    except ImportError:

        result["status"] = "unavailable"

        result["reasons"].append(
            "Pillow is not installed, so ELA could not be performed."
        )

        return result

    # ========================================================
    # FILE TYPE
    # ========================================================

    extension = os.path.splitext(
        image_path
    )[1].lower()

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    if extension not in supported_extensions:

        result["reasons"].append(
            "ELA requires a raster image. "
            "The supplied file is not a supported image format."
        )

        return result

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    if output_dir is None:

        output_dir = os.path.join(
            tempfile.gettempdir(),
            "doc_verify_ela",
        )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    unique_id = uuid.uuid4().hex

    resaved_path = os.path.join(
        output_dir,
        f"ela_resaved_{unique_id}.jpg",
    )

    ela_output_path = os.path.join(
        output_dir,
        f"ela_diff_{unique_id}.png",
    )

    # ========================================================
    # OPEN IMAGE
    # ========================================================

    try:

        with Image.open(image_path) as img:

            original = img.convert(
                "RGB"
            )

    except Exception as exc:

        result["status"] = "failed"

        result["reasons"].append(
            f"Could not open image for ELA: {exc}"
        )

        return result

    width, height = original.size

    result["details"]["width"] = width
    result["details"]["height"] = height
    result["details"]["ela_quality"] = ELA_QUALITY

    # ========================================================
    # MINIMUM SIZE
    # ========================================================

    if (
        width < MIN_IMAGE_WIDTH
        or height < MIN_IMAGE_HEIGHT
    ):

        result["status"] = "insufficient"

        result["reasons"].append(
            "Image is too small for reliable ELA analysis."
        )

        return result

    # ========================================================
    # JPEG RE-SAVE
    # ========================================================

    try:

        original.save(
            resaved_path,
            format="JPEG",
            quality=ELA_QUALITY,
        )

        with Image.open(resaved_path) as img:

            resaved = img.convert(
                "RGB"
            )

    except Exception as exc:

        result["status"] = "failed"

        result["reasons"].append(
            f"Could not create ELA comparison image: {exc}"
        )

        _safe_remove(
            resaved_path
        )

        return result

    # ========================================================
    # DIFFERENCE IMAGE
    # ========================================================

    try:

        diff = ImageChops.difference(
            original,
            resaved,
        )

        diff_array = np.asarray(
            diff,
            dtype=np.float32,
        )

    except Exception as exc:

        result["status"] = "failed"

        result["reasons"].append(
            f"ELA difference calculation failed: {exc}"
        )

        _safe_remove(
            resaved_path
        )

        return result

    # ========================================================
    # CONVERT RGB DIFFERENCE -> SINGLE ERROR MAP
    # ========================================================

    error_map = np.mean(
        diff_array,
        axis=2,
    )

    # ========================================================
    # GLOBAL STATISTICS
    # ========================================================

    global_mean = float(
        np.mean(error_map)
    )

    global_std = float(
        np.std(error_map)
    )

    maximum_difference = float(
        np.max(error_map)
    )

    p95 = float(
        np.percentile(
            error_map,
            95,
        )
    )

    p99 = float(
        np.percentile(
            error_map,
            99,
        )
    )

    result["details"].update(
        {
            "global_mean": round(
                global_mean,
                3,
            ),
            "global_std": round(
                global_std,
                3,
            ),
            "max_difference": round(
                maximum_difference,
                3,
            ),
            "p95_difference": round(
                p95,
                3,
            ),
            "p99_difference": round(
                p99,
                3,
            ),
        }
    )

    # ========================================================
    # LOCAL PATCH ANALYSIS
    # ========================================================

    patch_scores = _calculate_patch_scores(
        error_map,
        PATCH_SIZE,
    )

    if patch_scores:

        patch_array = np.asarray(
            patch_scores,
            dtype=np.float32,
        )

        suspicious_patch_count = int(
            np.sum(
                patch_array
                >= PATCH_MEAN_THRESHOLD
            )
        )

        total_patch_count = len(
            patch_scores
        )

        suspicious_patch_ratio = (
            suspicious_patch_count
            / total_patch_count
            if total_patch_count
            else 0.0
        )

        maximum_patch_mean = float(
            np.max(patch_array)
        )

        median_patch_mean = float(
            np.median(patch_array)
        )

    else:

        suspicious_patch_count = 0
        total_patch_count = 0
        suspicious_patch_ratio = 0.0
        maximum_patch_mean = 0.0
        median_patch_mean = 0.0

    result["details"].update(
        {
            "patch_size": PATCH_SIZE,

            "total_patches": (
                total_patch_count
            ),

            "suspicious_patches": (
                suspicious_patch_count
            ),

            "suspicious_patch_ratio": round(
                suspicious_patch_ratio,
                4,
            ),

            "maximum_patch_mean": round(
                maximum_patch_mean,
                3,
            ),

            "median_patch_mean": round(
                median_patch_mean,
                3,
            ),
        }
    )

    # ========================================================
    # SCORING
    # ========================================================

    score = 0
    reasons: List[str] = []

    # --------------------------------------------------------
    # Signal 1: unusually high global mean
    # --------------------------------------------------------

    if global_mean > GLOBAL_MEAN_THRESHOLD:

        score += 10

        reasons.append(
            "The image has elevated overall JPEG recompression error."
        )

    # --------------------------------------------------------
    # Signal 2: high global variance
    # --------------------------------------------------------

    if global_std > GLOBAL_STD_THRESHOLD:

        score += 15

        reasons.append(
            "ELA error levels vary substantially across the document."
        )

    # --------------------------------------------------------
    # Signal 3: upper-tail error
    # --------------------------------------------------------

    if p95 > P95_THRESHOLD:

        score += 10

        reasons.append(
            "A noticeable portion of the document has elevated "
            "compression differences."
        )

    if p99 > P99_THRESHOLD:

        score += 15

        reasons.append(
            "The highest-error regions show unusually strong "
            "compression differences."
        )

    # --------------------------------------------------------
    # Signal 4: localized suspicious regions
    # --------------------------------------------------------

    if (
        suspicious_patch_ratio
        > SUSPICIOUS_PATCH_RATIO_THRESHOLD
    ):

        score += 25

        reasons.append(
            f"{suspicious_patch_ratio * 100:.1f}% of analyzed "
            "image regions exceeded the local ELA threshold."
        )

    # --------------------------------------------------------
    # Signal 5:
    # One strong localized region while the median remains low.
    #
    # This pattern can be more interesting than globally noisy
    # scans because manipulation is often localized.
    # --------------------------------------------------------

    if (
        maximum_patch_mean
        > PATCH_MEAN_THRESHOLD * 1.5
        and median_patch_mean
        < PATCH_MEAN_THRESHOLD * 0.6
    ):

        score += 20

        reasons.append(
            "A localized region has much higher compression error "
            "than the typical region of the document."
        )

    # ========================================================
    # GENERATE HUMAN-REVIEW ELA IMAGE
    # ========================================================

    try:

        amplified = np.clip(
            diff_array
            * ELA_AMPLIFICATION,
            0,
            255,
        ).astype(
            np.uint8
        )

        Image.fromarray(
            amplified
        ).save(
            ela_output_path
        )

        result["ela_image_path"] = (
            ela_output_path
        )

    except Exception as exc:

        reasons.append(
            f"ELA statistics were calculated, but the visual "
            f"review image could not be saved: {exc}"
        )

    # ========================================================
    # CLEAN TEMP JPEG
    # ========================================================

    _safe_remove(
        resaved_path
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    result["checked"] = True
    result["status"] = "checked"

    result["score"] = min(
        100,
        round(
            score,
            1,
        ),
    )

    if reasons:

        result["reasons"] = reasons

    else:

        result["reasons"] = [
            "No significant compression-level anomalies "
            "were detected by ELA."
        ]

    return result


# ============================================================
# PATCH ANALYSIS
# ============================================================

def _calculate_patch_scores(
    error_map: np.ndarray,
    patch_size: int,
) -> List[float]:
    """
    Divide the ELA error map into patches and calculate the
    average ELA error for every sufficiently large patch.

    Local analysis is useful because tampering is often confined
    to a salary amount, name, date, or account field rather than
    affecting the entire document.
    """

    height, width = error_map.shape

    scores: List[float] = []

    for y in range(
        0,
        height,
        patch_size,
    ):

        for x in range(
            0,
            width,
            patch_size,
        ):

            patch = error_map[
                y:min(
                    y + patch_size,
                    height,
                ),
                x:min(
                    x + patch_size,
                    width,
                ),
            ]

            if patch.size == 0:
                continue

            patch_height, patch_width = (
                patch.shape
            )

            # Ignore tiny edge fragments.
            if (
                patch_height
                < patch_size // 2
                or patch_width
                < patch_size // 2
            ):
                continue

            scores.append(
                float(
                    np.mean(
                        patch
                    )
                )
            )

    return scores


# ============================================================
# SAFE FILE DELETE
# ============================================================

def _safe_remove(
    file_path: str,
) -> None:

    try:

        if os.path.exists(
            file_path
        ):
            os.remove(
                file_path
            )

    except OSError:

        pass
