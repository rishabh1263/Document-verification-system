"""
Common image tamper-analysis module.

This module provides conservative image-level manipulation signals for
document verification. It is reusable across PAN, Aadhaar, Voter ID,
Driving Licence, Passport, passbook and other document images.

IMPORTANT:
    This module does NOT prove that a document is fake.
    It produces evidence/risk signals that must be combined with OCR,
    layout, quality and (when available) authoritative verification.

Design goals:
    - CPU friendly
    - OpenCV + NumPy only
    - no OCR
    - no LLM
    - document-type independent
    - deterministic
    - conservative false-positive behavior
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_ANALYSIS_WIDTH = 1400

MIN_WIDTH = 300
MIN_HEIGHT = 180

ELA_JPEG_QUALITY = 90
ELA_RESIZE_WIDTH = 1000

BLOCK_SIZE = 32

LOW_RISK_THRESHOLD = 25
MEDIUM_RISK_THRESHOLD = 50
HIGH_RISK_THRESHOLD = 75

# Noise/texture signals are intentionally weak because normal documents
# naturally contain photos, text, signatures, seals and backgrounds.
NOISE_STD_DIFF_THRESHOLD = 34.0
NOISE_RANGE_THRESHOLD = 85.0

# ELA is supporting evidence only.
ELA_STD_THRESHOLD = 24.0
ELA_HIGH_ERROR_PERCENTAGE = 15.0

# Duplicate-region detection is deliberately strict.
DUPLICATE_SIMILARITY_THRESHOLD = 0.995

# A duplicated region must contain enough texture/edges to be meaningful.
MIN_PATCH_STD = 8.0
MIN_PATCH_EDGE_DENSITY = 1.5

# Adjacent/nearby blocks are not considered copy/paste evidence.
MIN_GRID_DISTANCE = 2


# ============================================================================
# RESULT HELPERS
# ============================================================================

def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "tamper_score": 0,
        "risk": "UNKNOWN",
        "decision": "MANUAL_REVIEW",
        "signals": [reason],
        "checks": {},
    }


def _risk_from_score(score: int) -> tuple[str, str]:
    if score >= HIGH_RISK_THRESHOLD:
        return "HIGH", "SUSPICIOUS"

    if score >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM", "REVIEW"

    return "LOW", "CLEAN"


# ============================================================================
# IMAGE DECODING
# ============================================================================

def decode_image(file_bytes: bytes) -> np.ndarray | None:
    if not file_bytes:
        return None

    try:
        array = np.frombuffer(
            file_bytes,
            dtype=np.uint8,
        )

        return cv2.imdecode(
            array,
            cv2.IMREAD_COLOR,
        )

    except Exception:
        return None


def prepare_image(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Image is None.")

    if image.size == 0:
        raise ValueError("Image is empty.")

    height, width = image.shape[:2]

    if width <= MAX_ANALYSIS_WIDTH:
        return image.copy()

    scale = MAX_ANALYSIS_WIDTH / width

    return cv2.resize(
        image,
        (
            MAX_ANALYSIS_WIDTH,
            max(1, int(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================================
# BASIC IMAGE INFORMATION
# ============================================================================

def image_statistics(image: np.ndarray) -> dict[str, Any]:
    if image is None or image.size == 0:
        return {
            "width": 0,
            "height": 0,
            "channels": 0,
            "aspect_ratio": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
        }

    height, width = image.shape[:2]

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
        channels = image.shape[2]
    else:
        gray = image
        channels = 1

    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "aspect_ratio": round(
            width / height,
            3,
        ) if height else 0.0,
        "brightness": round(
            float(np.mean(gray)),
            2,
        ),
        "contrast": round(
            float(np.std(gray)),
            2,
        ),
    }


# ============================================================================
# JPEG / ELA
# ============================================================================

def _jpeg_recompress(
    image: np.ndarray,
    quality: int = ELA_JPEG_QUALITY,
) -> np.ndarray:
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            int(quality),
        ],
    )

    if not success:
        raise ValueError("Could not JPEG-compress image.")

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    if decoded is None:
        raise ValueError("Could not decode recompressed image.")

    return decoded


def error_level_analysis(image: np.ndarray) -> dict[str, Any]:
    """
    Lightweight ELA-style analysis.

    ELA is NOT proof of editing. Text, edges, scans and JPEG history can
    naturally create elevated ELA values.
    """
    if image is None or image.size == 0:
        return {
            "available": False,
            "mean": 0.0,
            "std": 0.0,
            "max": 0.0,
            "high_error_percentage": 0.0,
        }

    working = image
    height, width = working.shape[:2]

    if width > ELA_RESIZE_WIDTH:
        scale = ELA_RESIZE_WIDTH / width
        working = cv2.resize(
            working,
            (
                ELA_RESIZE_WIDTH,
                max(1, int(height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    try:
        recompressed = _jpeg_recompress(working)

        difference = cv2.absdiff(
            working,
            recompressed,
        )

        gray_difference = cv2.cvtColor(
            difference,
            cv2.COLOR_BGR2GRAY,
        )

        mean_value = float(np.mean(gray_difference))
        std_value = float(np.std(gray_difference))
        max_value = float(np.max(gray_difference))

        high_threshold = max(
            20.0,
            mean_value + 2.0 * std_value,
        )

        high_percentage = float(
            np.mean(gray_difference > high_threshold) * 100.0
        )

        return {
            "available": True,
            "mean": round(mean_value, 2),
            "std": round(std_value, 2),
            "max": round(max_value, 2),
            "high_error_percentage": round(
                high_percentage,
                2,
            ),
        }

    except Exception as exc:
        return {
            "available": False,
            "mean": 0.0,
            "std": 0.0,
            "max": 0.0,
            "high_error_percentage": 0.0,
            "error": str(exc),
        }


# ============================================================================
# LOCAL NOISE / TEXTURE
# ============================================================================

def _block_std_values(
    gray: np.ndarray,
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    height, width = gray.shape[:2]
    values = []

    for y in range(
        0,
        height - block_size + 1,
        block_size,
    ):
        for x in range(
            0,
            width - block_size + 1,
            block_size,
        ):
            block = gray[
                y:y + block_size,
                x:x + block_size,
            ]

            values.append(float(np.std(block)))

    return np.asarray(
        values,
        dtype=np.float32,
    )


def noise_consistency_analysis(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    Analyze local texture/noise variation.

    The thresholds are deliberately conservative. Normal documents often
    have strong variation between photographs, text and background regions.
    """
    if image is None or image.size == 0:
        return {
            "available": False,
            "mean_block_std": 0.0,
            "std_of_block_std": 0.0,
            "range": 0.0,
            "blocks": 0,
        }

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    values = _block_std_values(gray)

    if values.size < 8:
        return {
            "available": False,
            "mean_block_std": 0.0,
            "std_of_block_std": 0.0,
            "range": 0.0,
            "blocks": int(values.size),
        }

    return {
        "available": True,
        "mean_block_std": round(float(np.mean(values)), 2),
        "std_of_block_std": round(float(np.std(values)), 2),
        "range": round(
            float(np.max(values) - np.min(values)),
            2,
        ),
        "blocks": int(values.size),
    }


# ============================================================================
# EDGE DISTRIBUTION
# ============================================================================

def edge_density_analysis(
    image: np.ndarray,
) -> dict[str, Any]:
    if image is None or image.size == 0:
        return {
            "available": False,
            "edge_density": 0.0,
            "block_std": 0.0,
        }

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    density = float(
        np.mean(edges > 0) * 100.0
    )

    values = _block_std_values(edges)

    block_std = (
        float(np.std(values))
        if values.size
        else 0.0
    )

    return {
        "available": True,
        "edge_density": round(density, 2),
        "block_std": round(block_std, 2),
    }


# ============================================================================
# DUPLICATE REGION DETECTION
# ============================================================================

def _small_gray(
    image: np.ndarray,
    size: tuple[int, int] = (32, 32),
) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = image

    return cv2.resize(
        gray,
        size,
        interpolation=cv2.INTER_AREA,
    )


def _normalized_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()

    if (
        float(np.std(a)) < 1e-6
        or float(np.std(b)) < 1e-6
    ):
        return 0.0

    correlation = np.corrcoef(a, b)[0, 1]

    if not np.isfinite(correlation):
        return 0.0

    return float(correlation)


def _patch_edge_density(
    gray_patch: np.ndarray,
) -> float:
    edges = cv2.Canny(
        gray_patch,
        50,
        150,
    )

    return float(
        np.mean(edges > 0) * 100.0
    )


def duplicate_region_analysis(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    Conservative duplicate/copy-paste signal.

    The old implementation incorrectly treated large uniform document
    regions as duplicate evidence. This version requires:

        1. very high similarity
        2. meaningful texture in BOTH regions
        3. meaningful edge content in BOTH regions
        4. sufficient spatial separation

    Therefore blank/background regions should no longer produce a duplicate
    signal merely because they are similar.
    """
    if image is None or image.size == 0:
        return {
            "available": False,
            "suspicious_pairs": 0,
            "max_similarity": 0.0,
        }

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    height, width = gray.shape[:2]

    grid_rows = 4
    grid_cols = 6

    patches = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            y1 = int(row * height / grid_rows)
            y2 = int((row + 1) * height / grid_rows)

            x1 = int(col * width / grid_cols)
            x2 = int((col + 1) * width / grid_cols)

            patch = gray[y1:y2, x1:x2]

            if patch.size == 0:
                continue

            small = _small_gray(patch)

            patch_std = float(np.std(small))
            edge_density = _patch_edge_density(small)

            # Ignore flat/background blocks.
            if patch_std < MIN_PATCH_STD:
                continue

            # Ignore blocks with almost no structural information.
            if edge_density < MIN_PATCH_EDGE_DENSITY:
                continue

            patches.append(
                {
                    "row": row,
                    "col": col,
                    "data": small,
                    "std": patch_std,
                    "edge_density": edge_density,
                }
            )

    suspicious_pairs = 0
    max_similarity = -1.0

    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            a = patches[i]
            b = patches[j]

            row_distance = abs(
                a["row"] - b["row"]
            )
            col_distance = abs(
                a["col"] - b["col"]
            )

            # Nearby regions naturally resemble each other.
            if (
                row_distance < MIN_GRID_DISTANCE
                and col_distance < MIN_GRID_DISTANCE
            ):
                continue

            similarity = _normalized_correlation(
                a["data"],
                b["data"],
            )

            max_similarity = max(
                max_similarity,
                similarity,
            )

            if similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
                suspicious_pairs += 1

    return {
        "available": True,
        "suspicious_pairs": suspicious_pairs,
        "max_similarity": round(
            max_similarity,
            4,
        ) if max_similarity >= 0 else 0.0,
    }


# ============================================================================
# SCORING
# ============================================================================

def _score_ela(
    ela: dict[str, Any],
) -> tuple[int, list[str]]:
    score = 0
    signals = []

    if not ela.get("available"):
        return score, signals

    std_value = float(ela.get("std", 0.0))
    high_percentage = float(
        ela.get(
            "high_error_percentage",
            0.0,
        )
    )

    # Require both signals before assigning meaningful ELA risk.
    if (
        std_value >= ELA_STD_THRESHOLD
        and high_percentage >= ELA_HIGH_ERROR_PERCENTAGE
    ):
        score += 12
        signals.append(
            "High and spatially distributed ELA variation detected."
        )

    return score, signals


def _score_noise(
    noise: dict[str, Any],
) -> tuple[int, list[str]]:
    score = 0
    signals = []

    if not noise.get("available"):
        return score, signals

    std_value = float(
        noise.get(
            "std_of_block_std",
            0.0,
        )
    )

    range_value = float(
        noise.get(
            "range",
            0.0,
        )
    )

    # One unusually variable metric is not enough.
    if (
        std_value >= NOISE_STD_DIFF_THRESHOLD
        and range_value >= NOISE_RANGE_THRESHOLD
    ):
        score += 8
        signals.append(
            "Strong local texture/noise inconsistency detected."
        )

    return score, signals


def _score_edges(
    edges: dict[str, Any],
) -> tuple[int, list[str]]:
    score = 0
    signals = []

    if not edges.get("available"):
        return score, signals

    block_std = float(
        edges.get(
            "block_std",
            0.0,
        )
    )

    edge_density = float(
        edges.get(
            "edge_density",
            0.0,
        )
    )

    # Weak supporting signal only.
    if (
        block_std >= 35.0
        and edge_density >= 2.0
    ):
        score += 4
        signals.append(
            "Uneven edge distribution detected."
        )

    return score, signals


def _score_duplicates(
    duplicate: dict[str, Any],
) -> tuple[int, list[str]]:
    score = 0
    signals = []

    if not duplicate.get("available"):
        return score, signals

    pairs = int(
        duplicate.get(
            "suspicious_pairs",
            0,
        )
    )

    # One pair is not enough to make a strong decision.
    if pairs >= 2:
        score += 18
        signals.append(
            "Multiple highly similar structured regions detected."
        )
    elif pairs == 1:
        score += 8
        signals.append(
            "One potentially duplicated structured region detected."
        )

    return score, signals


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_tampering(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    Analyze an image for possible manipulation.

    Interpretation:

        LOW / CLEAN
            No strong manipulation signal detected.

        MEDIUM / REVIEW
            Some suspicious image-level evidence exists.

        HIGH / SUSPICIOUS
            Multiple/strong image-level signals exist.

    None of these states proves document authenticity or forgery.
    """
    if image is None:
        return _empty_result("Image is None.")

    if image.size == 0:
        return _empty_result("Image is empty.")

    try:
        image = prepare_image(image)
    except Exception as exc:
        return _empty_result(
            f"Image preparation failed: {exc}"
        )

    height, width = image.shape[:2]

    if (
        width < MIN_WIDTH
        or height < MIN_HEIGHT
    ):
        return _empty_result(
            "Image is too small for reliable tamper analysis."
        )

    stats = image_statistics(image)
    ela = error_level_analysis(image)
    noise = noise_consistency_analysis(image)
    edges = edge_density_analysis(image)
    duplicate = duplicate_region_analysis(image)

    score = 0
    signals: list[str] = []

    for scorer, data in (
        (_score_ela, ela),
        (_score_noise, noise),
        (_score_edges, edges),
        (_score_duplicates, duplicate),
    ):
        component_score, component_signals = scorer(data)
        score += component_score
        signals.extend(component_signals)

    score = max(
        0,
        min(score, 100),
    )

    risk, decision = _risk_from_score(score)

    return {
        "tamper_score": int(score),
        "risk": risk,
        "decision": decision,
        "signals": signals,
        "checks": {
            "image": stats,
            "ela": ela,
            "noise_consistency": noise,
            "edge_distribution": edges,
            "duplicate_regions": duplicate,
        },
    }


def analyze_tampering_bytes(
    file_bytes: bytes,
) -> dict[str, Any]:
    image = decode_image(file_bytes)

    if image is None:
        return _empty_result(
            "Could not decode image."
        )

    return analyze_tampering(image)


def get_tamper_risk(
    image: np.ndarray,
) -> str:
    result = analyze_tampering(image)

    return str(
        result.get(
            "risk",
            "UNKNOWN",
        )
    )


# ============================================================================
# TEST / DIAGNOSTICS
# ============================================================================

def module_test() -> dict[str, Any]:
    """
    Deterministic smoke test.

    This test only verifies that the module executes and that the detector
    does not treat a simple synthetic document as high-risk.
    """
    image = np.full(
        (450, 702, 3),
        220,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (20, 20),
        (682, 430),
        (30, 30, 30),
        2,
    )

    cv2.putText(
        image,
        "INCOME TAX DEPARTMENT",
        (80, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        "ABCDE1234F",
        (80, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        "RAHUL KUMAR",
        (80, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    result = analyze_tampering(image)

    return {
        "passed": (
            isinstance(result, dict)
            and result.get("risk") != "HIGH"
        ),
        "result": result,
    }


def analyze_image_file(
    path: str,
) -> dict[str, Any]:
    """
    Convenience function for testing a local image file.
    """
    image = cv2.imread(path)

    if image is None:
        return _empty_result(
            f"Could not read image: {path}"
        )

    return analyze_tampering(image)


__all__ = [
    "decode_image",
    "prepare_image",
    "image_statistics",
    "error_level_analysis",
    "noise_consistency_analysis",
    "edge_density_analysis",
    "duplicate_region_analysis",
    "analyze_tampering",
    "analyze_tampering_bytes",
    "get_tamper_risk",
    "analyze_image_file",
    "module_test",
]