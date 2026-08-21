from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)


class ImageIntegrityService:

    def __init__(self):

        print("Loading AI Image Detector V2...")

        # ==========================================
        # 1. MODEL
        # ==========================================

        self.model_name = (
            "delpot/steganograph-ia-detector"
        )

        # ==========================================
        # 2. DEVICE
        # ==========================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ==========================================
        # 3. PROCESSOR
        # ==========================================

        self.processor = (
            AutoImageProcessor.from_pretrained(
                self.model_name
            )
        )

        # ==========================================
        # 4. MODEL
        # ==========================================

        self.model = (
            AutoModelForImageClassification
            .from_pretrained(
                self.model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        # ==========================================
        # 5. LABEL IDS
        # ==========================================

        self.real_label_id = None
        self.ai_label_id = None

        for index, label in (
            self.model.config.id2label.items()
        ):

            normalized = (
                str(label)
                .strip()
                .lower()
            )

            if normalized == "real":
                self.real_label_id = int(index)

            elif normalized == "ai_generated":
                self.ai_label_id = int(index)

        if self.real_label_id is None:
            raise ValueError(
                "AI detector does not contain "
                "a 'real' label."
            )

        if self.ai_label_id is None:
            raise ValueError(
                "AI detector does not contain "
                "an 'ai_generated' label."
            )

        # ==========================================
        # 6. THRESHOLD
        # ==========================================
        #
        # Initial conservative threshold.
        #
        # Do NOT assume this is permanently
        # calibrated from only two test images.
        # ==========================================

        self.synthetic_threshold = 0.90

        print(
            "AI Image Detector V2 Loaded Successfully "
            f"on {self.device}"
        )

        print(
            "AI detector labels:",
            self.model.config.id2label
        )


    def analyze(
        self,
        image_path: str
    ) -> dict:

        # ==========================================
        # 1. VALIDATE IMAGE
        # ==========================================

        path = Path(
            image_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # ==========================================
        # 2. LOAD IMAGE
        # ==========================================

        try:

            image = Image.open(
                path
            ).convert("RGB")

        except Exception as e:

            raise ValueError(
                f"Unable to read image: {e}"
            )

        # ==========================================
        # 3. PREPROCESS
        # ==========================================

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {

            key: value.to(
                self.device
            )

            for key, value
            in inputs.items()
        }

        # ==========================================
        # 4. INFERENCE
        # ==========================================

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

        # ==========================================
        # 5. SCORES
        # ==========================================

        real_score = float(
            probabilities[
                self.real_label_id
            ].item()
        )

        synthetic_score = float(
            probabilities[
                self.ai_label_id
            ].item()
        )

        # ==========================================
        # 6. SYNTHETIC DECISION
        # ==========================================

        synthetic_suspected = bool(
            synthetic_score
            >=
            self.synthetic_threshold
        )

        # ==========================================
        # 7. CLASSIFICATION
        # ==========================================

        if synthetic_suspected:

            classification = (
                "synthetic_suspected"
            )

        elif synthetic_score > real_score:

            classification = (
                "possible_synthetic_below_threshold"
            )

        else:

            classification = (
                "no_strong_synthetic_signal"
            )

        # ==========================================
        # 8. RESPONSE
        # ==========================================

        return {

            "synthetic_detection_available": True,

            "synthetic_suspected": (
                synthetic_suspected
            ),

            "classification": (
                classification
            ),

            "synthetic_score": round(
                synthetic_score,
                4
            ),

            "real_score": round(
                real_score,
                4
            ),

            "threshold": float(
                self.synthetic_threshold
            ),

            # ======================================
            # MANIPULATION DETECTION
            # ======================================

            "manipulation_detection_available": False,

            "manipulation_suspected": None,

            "manipulation_score": None
        }


image_integrity_service = (
    ImageIntegrityService()
)
