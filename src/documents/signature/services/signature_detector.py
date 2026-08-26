"""
YOLOS-based signature detection service.

Detects signature regions inside an image.

This service does NOT make the final
ACCEPT / REVIEW / REJECT decision.

Supported inputs:

    1. NumPy / OpenCV image
    2. PIL Image

YOLOS output is normalized to PIL RGB.

Duplicate overlapping detections are removed using:

    1. IoU-based NMS
    2. Containment-based suppression

The containment rule is important because YOLOS can return
a large loose box and a smaller precise box for the SAME
physical signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import cv2
import numpy as np
import torch

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
)


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = (
    "mdefrance/yolos-tiny-signature-detection"
)

DEFAULT_THRESHOLD = 0.50

NMS_IOU_THRESHOLD = 0.40

# If one box contains >= this percentage of another box,
# treat them as duplicate detections.
CONTAINMENT_THRESHOLD = 0.80


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# Supported image type
# =========================================================

ImageInput = Union[
    np.ndarray,
    Image.Image,
]


# =========================================================
# Detection structure
# =========================================================

@dataclass
class SignatureDetection:

    x1: int

    y1: int

    x2: int

    y2: int

    width: int

    height: int

    area: float

    area_ratio: float

    width_ratio: float

    height_ratio: float

    confidence: float

    image_width: int

    image_height: int

    # -----------------------------------------------------
    # Backward-compatible score property.
    # -----------------------------------------------------

    @property
    def score(self) -> float:

        return self.confidence


# =========================================================
# Detection result
# =========================================================

@dataclass
class SignatureDetectionResult:

    detected: bool

    detection_count: int

    highest_score: float

    largest_area_ratio: float

    multiple_signatures: bool

    detections: List[
        SignatureDetection
    ]


# =========================================================
# PIL conversion
# =========================================================

def _to_pil(
    image: ImageInput,
) -> Image.Image:
    """
    Normalize image to PIL RGB.
    """

    if image is None:

        raise ValueError(
            "Image cannot be None."
        )

    # -----------------------------------------------------
    # PIL
    # -----------------------------------------------------

    if isinstance(
        image,
        Image.Image,
    ):

        if (
            image.width <= 0
            or image.height <= 0
        ):

            raise ValueError(
                "Image dimensions must be positive."
            )

        return image.convert(
            "RGB"
        )

    # -----------------------------------------------------
    # NumPy
    # -----------------------------------------------------

    if isinstance(
        image,
        np.ndarray,
    ):

        if image.size == 0:

            raise ValueError(
                "Image cannot be empty."
            )

        # Grayscale
        if len(image.shape) == 2:

            return Image.fromarray(
                image
            ).convert("RGB")

        # Color
        if len(image.shape) == 3:

            channels = image.shape[2]

            # BGR
            if channels == 3:

                rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                )

                return Image.fromarray(
                    rgb
                )

            # BGRA
            if channels == 4:

                rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGRA2RGB,
                )

                return Image.fromarray(
                    rgb
                )

        raise ValueError(
            "Unsupported NumPy image format."
        )

    raise ValueError(
        "Image must be a NumPy array or PIL Image."
    )


# =========================================================
# Image validation
# =========================================================

def _validate_image(
    image: ImageInput,
) -> None:
    """
    Validate supported image types.
    """

    if image is None:

        raise ValueError(
            "Image cannot be None."
        )

    # -----------------------------------------------------
    # PIL
    # -----------------------------------------------------

    if isinstance(
        image,
        Image.Image,
    ):

        if (
            image.width <= 0
            or image.height <= 0
        ):

            raise ValueError(
                "Image dimensions must be positive."
            )

        return

    # -----------------------------------------------------
    # NumPy
    # -----------------------------------------------------

    if isinstance(
        image,
        np.ndarray,
    ):

        if image.size == 0:

            raise ValueError(
                "Image cannot be empty."
            )

        if len(image.shape) not in (
            2,
            3,
        ):

            raise ValueError(
                "Image must be grayscale or color."
            )

        height, width = (
            image.shape[:2]
        )

        if (
            height <= 0
            or width <= 0
        ):

            raise ValueError(
                "Image dimensions must be positive."
            )

        return

    raise ValueError(
        "Image must be a NumPy array or PIL Image."
    )


# =========================================================
# IoU
# =========================================================

def _calculate_iou(
    box_a: tuple[
        float,
        float,
        float,
        float,
    ],
    box_b: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:
    """
    Calculate Intersection over Union.

    Box:

        x1, y1, x2, y2
    """

    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(
        ax1,
        bx1,
    )

    intersection_y1 = max(
        ay1,
        by1,
    )

    intersection_x2 = min(
        ax2,
        bx2,
    )

    intersection_y2 = min(
        ay2,
        by2,
    )

    intersection_width = max(
        0.0,
        intersection_x2
        -
        intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2
        -
        intersection_y1,
    )

    intersection_area = (
        intersection_width
        *
        intersection_height
    )

    area_a = (
        max(
            0.0,
            ax2 - ax1,
        )
        *
        max(
            0.0,
            ay2 - ay1,
        )
    )

    area_b = (
        max(
            0.0,
            bx2 - bx1,
        )
        *
        max(
            0.0,
            by2 - by1,
        )
    )

    union_area = (
        area_a
        +
        area_b
        -
        intersection_area
    )

    if union_area <= 0:

        return 0.0

    return (
        intersection_area
        /
        union_area
    )


# =========================================================
# Box area
# =========================================================

def _box_area(
    box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:
    """
    Calculate box area.
    """

    x1, y1, x2, y2 = box

    return (
        max(
            0.0,
            x2 - x1,
        )
        *
        max(
            0.0,
            y2 - y1,
        )
    )


# =========================================================
# Intersection area
# =========================================================

def _intersection_area(
    box_a: tuple[
        float,
        float,
        float,
        float,
    ],
    box_b: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:
    """
    Calculate intersection area.
    """

    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b

    x1 = max(
        ax1,
        bx1,
    )

    y1 = max(
        ay1,
        by1,
    )

    x2 = min(
        ax2,
        bx2,
    )

    y2 = min(
        ay2,
        by2,
    )

    width = max(
        0.0,
        x2 - x1,
    )

    height = max(
        0.0,
        y2 - y1,
    )

    return (
        width
        *
        height
    )


# =========================================================
# Duplicate detection suppression
# =========================================================

def _non_maximum_suppression(
    detections: list[
        tuple[
            float,
            tuple[
                float,
                float,
                float,
                float,
            ],
        ]
    ],
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> list[
    tuple[
        float,
        tuple[
            float,
            float,
            float,
            float,
        ],
    ]
]:
    """
    Remove duplicate detections.

    Two mechanisms are used:

    1. IoU suppression
    2. Containment suppression

    Example:

        Detection A
        confidence = 0.913
        box = small precise signature

        Detection B
        confidence = 0.508
        box = large loose signature

    If B contains A, B is removed.

    This prevents one physical signature from becoming
    MULTIPLE_SIGNATURES.
    """

    if not detections:

        return []

    if not (
        0.0
        <= iou_threshold
        <= 1.0
    ):

        raise ValueError(
            "iou_threshold must be between "
            "0.0 and 1.0."
        )

    # -----------------------------------------------------
    # Strongest first.
    # -----------------------------------------------------

    ordered = sorted(
        detections,
        key=lambda item: item[0],
        reverse=True,
    )

    kept = []

    # -----------------------------------------------------
    # Process each detection.
    # -----------------------------------------------------

    for (
        candidate_score,
        candidate_box,
    ) in ordered:

        candidate_area = _box_area(
            candidate_box
        )

        if candidate_area <= 0:

            continue

        should_keep = True

        # -------------------------------------------------
        # Compare candidate with stronger detections.
        # -------------------------------------------------

        for (
            kept_score,
            kept_box,
        ) in kept:

            kept_area = _box_area(
                kept_box
            )

            if kept_area <= 0:

                continue

            # =============================================
            # Rule 1: IoU
            # =============================================

            iou = _calculate_iou(
                candidate_box,
                kept_box,
            )

            if iou >= iou_threshold:

                should_keep = False

                break

            # =============================================
            # Rule 2: Containment
            # =============================================

            intersection = (
                _intersection_area(
                    candidate_box,
                    kept_box,
                )
            )

            if intersection <= 0:

                continue

            candidate_containment = (
                intersection
                /
                candidate_area
            )

            kept_containment = (
                intersection
                /
                kept_area
            )

            # -------------------------------------------------
            # Candidate is mostly inside stronger box.
            # -------------------------------------------------

            if (
                candidate_containment
                >= CONTAINMENT_THRESHOLD
            ):

                should_keep = False

                break

            # -------------------------------------------------
            # Stronger box is mostly inside candidate.
            #
            # Candidate is probably a large loose box.
            # Since kept_score >= candidate_score, keep
            # the stronger precise detection.
            # -------------------------------------------------

            if (
                kept_containment
                >= CONTAINMENT_THRESHOLD
            ):

                should_keep = False

                break

        if should_keep:

            kept.append(
                (
                    candidate_score,
                    candidate_box,
                )
            )

    return kept


# =========================================================
# Singleton
# =========================================================

_detector = None


# =========================================================
# Detector
# =========================================================

class SignatureDetector:
    """
    YOLOS signature detector.

    Model is loaded once per process.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        threshold: float = DEFAULT_THRESHOLD,
        nms_iou_threshold: float = NMS_IOU_THRESHOLD,
    ):

        self.model_name = model_name

        self.threshold = float(
            threshold
        )

        self.nms_iou_threshold = float(
            nms_iou_threshold
        )

        # -------------------------------------------------
        # Threshold validation.
        # -------------------------------------------------

        if not (
            0.0
            <= self.threshold
            <= 1.0
        ):

            raise ValueError(
                "threshold must be between "
                "0.0 and 1.0."
            )

        if not (
            0.0
            <= self.nms_iou_threshold
            <= 1.0
        ):

            raise ValueError(
                "nms_iou_threshold must be between "
                "0.0 and 1.0."
            )

        self.device = torch.device(
            DEVICE
        )

        # -------------------------------------------------
        # Processor.
        # -------------------------------------------------

        self.processor = (
            AutoImageProcessor.from_pretrained(
                self.model_name
            )
        )

        # -------------------------------------------------
        # YOLOS.
        # -------------------------------------------------

        self.model = (
            AutoModelForObjectDetection
            .from_pretrained(
                self.model_name
            )
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

    # =====================================================
    # Predict
    # =====================================================

    def predict(
        self,
        image: ImageInput,
    ) -> SignatureDetectionResult:
        """
        Detect signature regions.
        """

        # -------------------------------------------------
        # Validate.
        # -------------------------------------------------

        _validate_image(
            image
        )

        # -------------------------------------------------
        # Convert to PIL RGB.
        # -------------------------------------------------

        pil_image = _to_pil(
            image
        )

        width = pil_image.width

        height = pil_image.height

        image_area = (
            width
            *
            height
        )

        if image_area <= 0:

            return SignatureDetectionResult(
                detected=False,
                detection_count=0,
                highest_score=0.0,
                largest_area_ratio=0.0,
                multiple_signatures=False,
                detections=[],
            )

        # -------------------------------------------------
        # Processor.
        # -------------------------------------------------

        inputs = self.processor(
            images=pil_image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value in inputs.items()
        }

        # -------------------------------------------------
        # Inference.
        # -------------------------------------------------

        with torch.inference_mode():

            outputs = self.model(
                **inputs
            )

        # -------------------------------------------------
        # Post-processing.
        # -------------------------------------------------

        target_sizes = torch.tensor(
            [
                [
                    height,
                    width,
                ]
            ],
            device=self.device,
        )

        results = (
            self.processor
            .post_process_object_detection(
                outputs,
                threshold=self.threshold,
                target_sizes=target_sizes,
            )
        )

        result = results[0]

        scores = result["scores"]

        boxes = result["boxes"]

        # -------------------------------------------------
        # Collect raw detections.
        # -------------------------------------------------

        raw_detections = []

        for index in range(
            len(scores)
        ):

            score = float(
                scores[index].item()
            )

            box = (
                boxes[index]
                .detach()
                .cpu()
                .numpy()
            )

            if box.size != 4:

                continue

            x1, y1, x2, y2 = (
                float(value)
                for value in box
            )

            # -------------------------------------------------
            # Clamp.
            # -------------------------------------------------

            x1 = max(
                0.0,
                min(
                    x1,
                    float(width),
                ),
            )

            y1 = max(
                0.0,
                min(
                    y1,
                    float(height),
                ),
            )

            x2 = max(
                0.0,
                min(
                    x2,
                    float(width),
                ),
            )

            y2 = max(
                0.0,
                min(
                    y2,
                    float(height),
                ),
            )

            # -------------------------------------------------
            # Correct coordinate order.
            # -------------------------------------------------

            if x2 < x1:

                x1, x2 = (
                    x2,
                    x1,
                )

            if y2 < y1:

                y1, y2 = (
                    y2,
                    y1,
                )

            box_width = (
                x2 - x1
            )

            box_height = (
                y2 - y1
            )

            if (
                box_width <= 0
                or box_height <= 0
            ):

                continue

            raw_detections.append(
                (
                    score,
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                )
            )

        # -------------------------------------------------
        # No detections.
        # -------------------------------------------------

        if not raw_detections:

            return SignatureDetectionResult(
                detected=False,
                detection_count=0,
                highest_score=0.0,
                largest_area_ratio=0.0,
                multiple_signatures=False,
                detections=[],
            )

        # -------------------------------------------------
        # Remove duplicate boxes.
        # -------------------------------------------------

        filtered_detections = (
            _non_maximum_suppression(
                raw_detections,
                iou_threshold=(
                    self.nms_iou_threshold
                ),
            )
        )

        # -------------------------------------------------
        # Convert into dataclass objects.
        # -------------------------------------------------

        detections: List[
            SignatureDetection
        ] = []

        for (
            score,
            box,
        ) in filtered_detections:

            x1, y1, x2, y2 = box

            box_width = (
                x2 - x1
            )

            box_height = (
                y2 - y1
            )

            area = (
                box_width
                *
                box_height
            )

            area_ratio = (
                area
                /
                image_area
            )

            width_ratio = (
                box_width
                /
                width
            )

            height_ratio = (
                box_height
                /
                height
            )

            detection = (
                SignatureDetection(

                    x1=int(
                        round(x1)
                    ),

                    y1=int(
                        round(y1)
                    ),

                    x2=int(
                        round(x2)
                    ),

                    y2=int(
                        round(y2)
                    ),

                    width=int(
                        round(
                            box_width
                        )
                    ),

                    height=int(
                        round(
                            box_height
                        )
                    ),

                    area=round(
                        area,
                        2,
                    ),

                    area_ratio=round(
                        area_ratio,
                        6,
                    ),

                    width_ratio=round(
                        width_ratio,
                        6,
                    ),

                    height_ratio=round(
                        height_ratio,
                        6,
                    ),

                    confidence=round(
                        score,
                        6,
                    ),

                    image_width=width,

                    image_height=height,
                )
            )

            detections.append(
                detection
            )

        # -------------------------------------------------
        # Safety.
        # -------------------------------------------------

        if not detections:

            return SignatureDetectionResult(
                detected=False,
                detection_count=0,
                highest_score=0.0,
                largest_area_ratio=0.0,
                multiple_signatures=False,
                detections=[],
            )

        # -------------------------------------------------
        # Strongest first.
        # -------------------------------------------------

        detections.sort(
            key=lambda detection:
                detection.confidence,
            reverse=True,
        )

        # -------------------------------------------------
        # Highest confidence.
        # -------------------------------------------------

        highest_score = max(
            detection.confidence
            for detection in detections
        )

        # -------------------------------------------------
        # Largest area ratio.
        # -------------------------------------------------

        largest_area_ratio = max(
            detection.area_ratio
            for detection in detections
        )

        # -------------------------------------------------
        # Multiple signatures.
        #
        # IMPORTANT:
        # At this point duplicate overlapping boxes have
        # already been removed.
        # -------------------------------------------------

        multiple_signatures = (
            len(detections) > 1
        )

        return SignatureDetectionResult(

            detected=True,

            detection_count=len(
                detections
            ),

            highest_score=round(
                highest_score,
                6,
            ),

            largest_area_ratio=round(
                largest_area_ratio,
                6,
            ),

            multiple_signatures=(
                multiple_signatures
            ),

            detections=detections,
        )


# =========================================================
# Singleton getter
# =========================================================

def get_signature_detector() -> SignatureDetector:
    """
    Return reusable detector instance.

    YOLOS is loaded only once per process.
    """

    global _detector

    if _detector is None:

        _detector = (
            SignatureDetector()
        )

    return _detector


# =========================================================
# Convenience function
# =========================================================

def detect_signatures(
    image: ImageInput,
    threshold: Optional[float] = None,
) -> SignatureDetectionResult:
    """
    Detect signatures.

    Supports:

        PIL.Image
        NumPy ndarray
    """

    detector = (
        get_signature_detector()
    )

    # -----------------------------------------------------
    # Default threshold.
    # -----------------------------------------------------

    if threshold is None:

        return detector.predict(
            image
        )

    # -----------------------------------------------------
    # Validate temporary threshold.
    # -----------------------------------------------------

    threshold = float(
        threshold
    )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):

        raise ValueError(
            "threshold must be between "
            "0.0 and 1.0."
        )

    original_threshold = (
        detector.threshold
    )

    try:

        detector.threshold = (
            threshold
        )

        return detector.predict(
            image
        )

    finally:

        detector.threshold = (
            original_threshold
        )