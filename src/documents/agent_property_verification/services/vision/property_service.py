from PIL import Image
import torch
import open_clip

from src.documents.agent_property_verification.services.vision.property_crop_service import property_crop_service


class PropertyService:

    def __init__(self):

        print("Loading Property Classification Model...")

        # ==========================================
        # 1. DEVICE
        # ==========================================

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ==========================================
        # 2. LOAD CLIP MODEL
        # ==========================================

        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="laion2b_s34b_b79k"
            )
        )

        self.model.to(self.device)
        self.model.eval()

        # ==========================================
        # 3. PROPERTY EXISTENCE PROMPTS
        # ==========================================
        #
        # These prompts determine whether a real
        # property/building is visible.
        #
        # They intentionally cover:
        # - normal houses
        # - small houses
        # - partially visible houses
        # - apartments
        # - commercial buildings
        # - unfinished properties
        # - construction sites
        # - agent selfies outside properties
        # ==========================================

        self.property_detection_prompts = [

            # --------------------------------------
            # GENERAL PROPERTY
            # --------------------------------------

            "a clearly visible residential property exterior",

            "a clearly visible house exterior",

            "a real estate property visible in the background",

            "a residential building visible in the background",

            "an exterior view of a residential property",

            "a building exterior visible in an outdoor photograph",

            # --------------------------------------
            # SMALL / SIMPLE RESIDENTIAL PROPERTY
            # --------------------------------------

            "a small residential house visible behind a person",

            "a simple residential house exterior",

            "a modest residential property exterior",

            "a small single-storey house",

            "a small residential building",

            "a partially visible residential house",

            "a house partially visible in the background",

            "a residential structure visible behind a person",

            "a small home visible in the background",

            "a simple house visible behind a person",

            # --------------------------------------
            # HOUSE ENTRANCE / PARTIAL STRUCTURE
            # --------------------------------------

            "a residential entrance with a roof and doorway",

            "a small home with a covered entrance",

            "a house entrance visible behind a person",

            "part of a residential house exterior",

            "a residential doorway and exterior structure",

            "a house with a roof and entrance visible",

            # --------------------------------------
            # AGENT + PROPERTY
            # --------------------------------------

            "a person taking a selfie in front of a building",

            "a person standing outside a property",

            "a person taking a selfie outside a house",

            "a person taking a selfie outside a small house",

            "a person standing in front of a residential house",

            "a person with a residential property behind them",

            "a field agent taking a selfie at a property",

            # --------------------------------------
            # APARTMENT / LARGE BUILDING
            # --------------------------------------

            "a clearly visible apartment building exterior",

            "a multi-storey residential building",

            "a large building visible behind a person",

            "an apartment building visible behind a person",

            # --------------------------------------
            # COMMERCIAL PROPERTY
            # --------------------------------------

            "a clearly visible commercial building exterior",

            "a commercial office building exterior",

            "a retail property exterior",

            "a warehouse building exterior",

            # --------------------------------------
            # UNDER CONSTRUCTION PROPERTY
            # --------------------------------------

            "an exterior view of a building under construction",

            "an unfinished concrete building under construction",

            "a residential construction site with a building",

            "a partially completed house",

            "a partially completed residential building",

            "a multi-storey concrete building under construction",

            "a construction site showing the structure of a building",

            "a building with scaffolding",

            "an unfinished property with exposed concrete",

            "a house under construction",

            "an unfinished residential property",

            # --------------------------------------
            # ARCHITECTURAL STRUCTURE
            # --------------------------------------

            "a property exterior with windows doors walls or balconies",

            "a large architectural structure occupying the background",

            "a residential structure with walls roof and doorway",

            "an outdoor photograph containing a visible building"
        ]

        # ==========================================
        # 4. NON-PROPERTY PROMPTS
        # ==========================================
        #
        # These represent images that should NOT
        # pass property verification.
        # ==========================================

        self.non_property_prompts = [

            "a close-up portrait with no building visible",

            "a selfie where only the person's face is visible",

            "an indoor selfie inside a room",

            "a person standing inside a room",

            "an indoor room with furniture",

            "an indoor room with walls and furniture",

            "a landscape with no building visible",

            "an empty road with no property visible",

            "an outdoor selfie with only trees and sky",

            "an outdoor selfie with only vegetation",

            "a person standing outside with only trees behind them",

            "an outdoor photograph with only road and vegetation",

            "an outdoor photo with no architectural structure",

            "a photo where the background contains no building",

            "a close-up person with no property visible",

            "a photo without a clearly visible property",

            "a natural landscape without buildings",

            "an empty field without buildings",

            "a street scene where no property is visible"
        ]

        # ==========================================
        # 5. PROPERTY TYPE PROMPTS
        # ==========================================

        self.property_prompts = [

            "an exterior view of a residential house",

            "an exterior view of an apartment or residential building",

            "an exterior view of a commercial office building",

            "an exterior view of a warehouse or industrial building",

            "an exterior view of a retail shop",

            "a residential building under construction",

            "a commercial building under construction"
        ]

        self.property_labels = [

            "Residential House",

            "Apartment",

            "Commercial Building",

            "Warehouse",

            "Retail Shop",

            "Residential Under Construction",

            "Commercial Under Construction"
        ]

        # ==========================================
        # 6. TOKENIZER
        # ==========================================

        tokenizer = open_clip.get_tokenizer(
            "ViT-B-32"
        )

        # ==========================================
        # 7. PROPERTY DETECTION TEXT FEATURES
        # ==========================================

        detection_prompts = (
            self.property_detection_prompts
            +
            self.non_property_prompts
        )

        with torch.no_grad():

            detection_text = tokenizer(
                detection_prompts
            ).to(self.device)

            self.detection_features = (
                self.model.encode_text(
                    detection_text
                )
            )

            self.detection_features /= (
                self.detection_features.norm(
                    dim=-1,
                    keepdim=True
                )
            )

        # ==========================================
        # 8. PROPERTY TYPE TEXT FEATURES
        # ==========================================

        with torch.no_grad():

            property_text = tokenizer(
                self.property_prompts
            ).to(self.device)

            self.property_features = (
                self.model.encode_text(
                    property_text
                )
            )

            self.property_features /= (
                self.property_features.norm(
                    dim=-1,
                    keepdim=True
                )
            )

        print("Property Classification Model Loaded")


    def classify(self, image_path: str) -> dict:

        # ==========================================
        # 1. PREPARE PROPERTY IMAGE
        # ==========================================

        image = property_crop_service.crop(
            image_path
        )

        if not isinstance(image, Image.Image):

            raise ValueError(
                "Property crop service must return a PIL image."
            )

        # ==========================================
        # 2. PREPROCESS IMAGE
        # ==========================================

        image_tensor = (
            self.preprocess(image)
            .unsqueeze(0)
            .to(self.device)
        )

        # ==========================================
        # 3. IMAGE EMBEDDING
        # ==========================================

        with torch.no_grad():

            image_features = (
                self.model.encode_image(
                    image_tensor
                )
            )

            image_features /= (
                image_features.norm(
                    dim=-1,
                    keepdim=True
                )
            )

        # ==========================================
        # 4. PROPERTY VS NON-PROPERTY SIMILARITY
        # ==========================================

        with torch.no_grad():

            similarity = (
                image_features
                @ self.detection_features.T
            )[0]

        property_count = len(
            self.property_detection_prompts
        )

        property_scores = similarity[
            :property_count
        ]

        non_property_scores = similarity[
            property_count:
        ]

        # ==========================================
        # 5. TOP-K GROUP SCORING
        # ==========================================
        #
        # Average several strong prompts instead
        # of trusting one individual prompt.
        # ==========================================

        property_top_k = min(
            3,
            len(property_scores)
        )

        non_property_top_k = min(
            3,
            len(non_property_scores)
        )

        best_property_scores = torch.topk(
            property_scores,
            property_top_k
        ).values

        best_non_property_scores = torch.topk(
            non_property_scores,
            non_property_top_k
        ).values

        property_score = float(
            best_property_scores.mean().item()
        )

        non_property_score = float(
            best_non_property_scores.mean().item()
        )

        margin = (
            property_score
            -
            non_property_score
        )

        # ==========================================
        # 6. PROPERTY DECISION
        # ==========================================
        #
        # DO NOT loosen this yet.
        #
        # We first improve semantic prompts and
        # evaluate positive + negative test images.
        # ==========================================

        PROPERTY_MARGIN = -0.005

        property_detected = bool(
            margin > PROPERTY_MARGIN
        )

        # ==========================================
        # 7. DEBUG INFORMATION
        # ==========================================

        best_property_index = int(
            property_scores.argmax().item()
        )

        best_non_property_index = int(
            non_property_scores.argmax().item()
        )

        debug = {

            "property_score": round(
                property_score,
                4
            ),

            "non_property_score": round(
                non_property_score,
                4
            ),

            "margin": round(
                margin,
                4
            ),

            "best_property_prompt": (
                self.property_detection_prompts[
                    best_property_index
                ]
            ),

            "best_non_property_prompt": (
                self.non_property_prompts[
                    best_non_property_index
                ]
            )
        }

        # ==========================================
        # 8. NO PROPERTY
        # ==========================================

        if not property_detected:

            return {

                "detected": False,

                "type": None,

                "debug": debug
            }

        # ==========================================
        # 9. PROPERTY TYPE CLASSIFICATION
        # ==========================================

        with torch.no_grad():

            type_similarity = (
                image_features
                @ self.property_features.T
            )[0]

        best_index = int(
            type_similarity.argmax().item()
        )

        property_type = (
            self.property_labels[
                best_index
            ]
        )

        # ==========================================
        # 10. TYPE DEBUG
        # ==========================================

        debug["type_score"] = round(
            float(
                type_similarity[
                    best_index
                ].item()
            ),
            4
        )

        # ==========================================
        # 11. FINAL RESULT
        # ==========================================

        return {

            "detected": True,

            "type": property_type,

            "debug": debug
        }


property_service = PropertyService()
