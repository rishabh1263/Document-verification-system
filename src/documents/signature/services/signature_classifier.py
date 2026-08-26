"""
MobileNetV3 signature classifier.

Loads the trained signature/non-signature classifier and
provides reusable inference for the signature validation
pipeline.

Supported input types:

    1. NumPy / OpenCV image
    2. PIL Image

All inputs are normalized to RGB before inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


# =========================================================
# Configuration
# =========================================================

MODEL_PATH = Path(
    "src/documents/signature/models/signature_classifier.pth"
)

IMAGE_SIZE = 224

DEFAULT_SIGNATURE_THRESHOLD = 0.90


# =========================================================
# Result
# =========================================================

@dataclass
class SignatureClassifierResult:

    predicted_class: str

    signature_probability: float

    non_signature_probability: float

    confidence: float

    is_signature: bool


# =========================================================
# Supported image type
# =========================================================

ImageInput = Union[
    np.ndarray,
    Image.Image,
]


# =========================================================
# Classifier
# =========================================================

class SignatureClassifier:
    """
    MobileNetV3-based binary signature classifier.

    Expected classes:

        0 -> non_signature
        1 -> signature

    The actual class ordering is read from the checkpoint
    rather than assumed blindly.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        threshold: float = DEFAULT_SIGNATURE_THRESHOLD,
    ) -> None:

        self.model_path = Path(
            model_path
        )

        self.threshold = float(
            threshold
        )

        if not (
            0.0
            <= self.threshold
            <= 1.0
        ):

            raise ValueError(
                "threshold must be between "
                "0.0 and 1.0."
            )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.classes: list[str] = []

        self.model = None

        self.transform = (
            self._create_transform()
        )

        self._load_model()

    # =====================================================
    # Transform
    # =====================================================

    @staticmethod
    def _create_transform():

        return transforms.Compose(
            [

                transforms.Resize(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE,
                    )
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )

    # =====================================================
    # Load model
    # =====================================================

    def _load_model(self) -> None:

        if not self.model_path.exists():

            raise FileNotFoundError(
                "Signature classifier model "
                f"not found: {self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        if not isinstance(
            checkpoint,
            dict,
        ):

            raise RuntimeError(
                "Invalid signature classifier "
                "checkpoint format."
            )

        if "classes" not in checkpoint:

            raise RuntimeError(
                "Classifier checkpoint does not "
                "contain 'classes'."
            )

        if "model_state_dict" not in checkpoint:

            raise RuntimeError(
                "Classifier checkpoint does not "
                "contain 'model_state_dict'."
            )

        self.classes = list(
            checkpoint["classes"]
        )

        # -------------------------------------------------
        # Validate class structure.
        # -------------------------------------------------

        if "signature" not in self.classes:

            raise RuntimeError(
                "Model does not contain "
                "'signature' class."
            )

        if "non_signature" not in self.classes:

            raise RuntimeError(
                "Model does not contain "
                "'non_signature' class."
            )

        if len(self.classes) != 2:

            raise RuntimeError(
                "Signature classifier must contain "
                "exactly two classes."
            )

        # -------------------------------------------------
        # Build MobileNetV3 Small.
        # -------------------------------------------------

        self.model = (
            models.mobilenet_v3_small(
                weights=None
            )
        )

        in_features = (
            self.model
            .classifier[-1]
            .in_features
        )

        self.model.classifier[-1] = (
            torch.nn.Linear(
                in_features,
                len(self.classes),
            )
        )

        # -------------------------------------------------
        # Load trained weights.
        # -------------------------------------------------

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

    # =====================================================
    # PIL conversion
    # =====================================================

    @staticmethod
    def _to_pil(
        image: ImageInput,
    ) -> Image.Image:
        """
        Convert supported image input to RGB PIL.

        PIL:
            Any PIL mode is converted to RGB.

        NumPy:
            2D       -> grayscale -> RGB
            3-channel -> BGR -> RGB
            4-channel -> BGRA -> RGB
        """

        if image is None:

            raise ValueError(
                "Image cannot be None."
            )

        # -------------------------------------------------
        # PIL image
        # -------------------------------------------------

        if isinstance(
            image,
            Image.Image,
        ):

            if (
                image.width <= 0
                or
                image.height <= 0
            ):

                raise ValueError(
                    "Image dimensions must be positive."
                )

            # IMPORTANT:
            #
            # This handles:
            #
            # P
            # L
            # LA
            # RGB
            # RGBA
            # CMYK
            # etc.
            #
            # consistently.
            return image.convert(
                "RGB"
            )

        # -------------------------------------------------
        # NumPy image
        # -------------------------------------------------

        if not isinstance(
            image,
            np.ndarray,
        ):

            raise ValueError(
                "Image must be a NumPy array "
                "or PIL Image."
            )

        if image.size == 0:

            raise ValueError(
                "Image cannot be empty."
            )

        # -------------------------------------------------
        # Grayscale
        # -------------------------------------------------

        if len(image.shape) == 2:

            pil_image = Image.fromarray(
                image
            )

            return pil_image.convert(
                "RGB"
            )

        # -------------------------------------------------
        # Color
        # -------------------------------------------------

        if len(image.shape) == 3:

            channels = image.shape[2]

            # BGR
            if channels == 3:

                # OpenCV -> RGB
                rgb = image[
                    ...,
                    ::-1
                ].copy()

                return Image.fromarray(
                    rgb
                ).convert(
                    "RGB"
                )

            # BGRA
            if channels == 4:

                # Ignore alpha and convert
                # BGR -> RGB.
                rgb = image[
                    ...,
                    :3
                ][
                    ...,
                    ::-1
                ].copy()

                return Image.fromarray(
                    rgb
                ).convert(
                    "RGB"
                )

        raise ValueError(
            "Unsupported NumPy image format."
        )

    # =====================================================
    # Prediction
    # =====================================================

    def predict(
        self,
        image: ImageInput,
    ) -> SignatureClassifierResult:
        """
        Run signature classification.

        Returns probabilities for:

            signature
            non_signature
        """

        # -------------------------------------------------
        # Normalize input.
        # -------------------------------------------------

        pil_image = self._to_pil(
            image
        )

        # -------------------------------------------------
        # Transform.
        # -------------------------------------------------

        tensor = self.transform(
            pil_image
        )

        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            self.device
        )

        # -------------------------------------------------
        # Inference.
        # -------------------------------------------------

        with torch.inference_mode():

            output = self.model(
                tensor
            )

            probabilities = (
                torch.softmax(
                    output,
                    dim=1,
                )[0]
            )

        # -------------------------------------------------
        # Map probabilities using checkpoint classes.
        # -------------------------------------------------

        probability_map = {

            self.classes[index]:
                float(
                    probabilities[index]
                    .item()
                )

            for index in range(
                len(self.classes)
            )
        }

        signature_probability = (
            probability_map.get(
                "signature",
                0.0,
            )
        )

        non_signature_probability = (
            probability_map.get(
                "non_signature",
                0.0,
            )
        )

        # -------------------------------------------------
        # Decision.
        # -------------------------------------------------

        if (
            signature_probability
            >= self.threshold
        ):

            predicted_class = (
                "signature"
            )

            confidence = (
                signature_probability
            )

            is_signature = True

        else:

            predicted_class = (
                "non_signature"
            )

            confidence = (
                non_signature_probability
            )

            is_signature = False

        return SignatureClassifierResult(

            predicted_class=(
                predicted_class
            ),

            signature_probability=round(
                signature_probability,
                6,
            ),

            non_signature_probability=round(
                non_signature_probability,
                6,
            ),

            confidence=round(
                confidence,
                6,
            ),

            is_signature=(
                is_signature
            ),
        )


# =========================================================
# Singleton
# =========================================================

_classifier: Optional[
    SignatureClassifier
] = None


def get_signature_classifier() -> SignatureClassifier:
    """
    Return a reusable classifier instance.

    The model is loaded only once per process.
    """

    global _classifier

    if _classifier is None:

        _classifier = (
            SignatureClassifier()
        )

    return _classifier


# =========================================================
# Convenience function
# =========================================================

def classify_signature(
    image: ImageInput,
) -> SignatureClassifierResult:
    """
    Classify an image as signature/non-signature.

    Supports:

        PIL.Image
        NumPy ndarray
    """

    classifier = (
        get_signature_classifier()
    )

    return classifier.predict(
        image
    )