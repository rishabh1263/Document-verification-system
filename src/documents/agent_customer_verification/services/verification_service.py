"""
Agent / Customer Face Verification Service.

Compares detected InsightFace embeddings against the stored
Agent and Customer reference embeddings.
"""

from pathlib import Path

import numpy as np


class VerificationService:

    SIMILARITY_THRESHOLD = 0.55

    def __init__(self):

        # Always resolve relative to THIS module.
        # This prevents Uvicorn's working directory from affecting
        # where embeddings are loaded from.
        module_root = Path(__file__).resolve().parents[1]

        self.embedding_dir = (
            module_root / "embeddings"
        )

        self.threshold = self.SIMILARITY_THRESHOLD

    # ========================================================
    # LOAD EMBEDDING
    # ========================================================

    def load_embedding(
        self,
        filename: str,
    ) -> np.ndarray:

        path = self.embedding_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Embedding file not found: {path}"
            )

        embedding = np.load(
            path,
            allow_pickle=False,
        )

        if embedding.size == 0:
            raise ValueError(
                f"Embedding file is empty: {path}"
            )

        return embedding

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    @staticmethod
    def cosine_similarity(
        emb1,
        emb2,
    ) -> float:

        emb1 = np.asarray(
            emb1,
            dtype=np.float32,
        ).reshape(-1)

        emb2 = np.asarray(
            emb2,
            dtype=np.float32,
        ).reshape(-1)

        if emb1.shape != emb2.shape:
            raise ValueError(
                "Embedding dimensions do not match: "
                f"{emb1.shape} vs {emb2.shape}"
            )

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError(
                "Cannot compare zero-length face embedding."
            )

        emb1 = emb1 / norm1
        emb2 = emb2 / norm2

        return float(
            np.dot(
                emb1,
                emb2,
            )
        )

    # ========================================================
    # VERIFY AGENT + CUSTOMER
    # ========================================================

    def verify_group(
        self,
        detected_faces,
    ):

        if len(detected_faces) != 2:

            return {
                "success": False,
                "message": (
                    "Exactly two faces are required."
                ),
            }

        # ====================================================
        # LOAD REFERENCE EMBEDDINGS
        # ====================================================

        agent_embedding = self.load_embedding(
            "agent.npy"
        )

        customer_embedding = self.load_embedding(
            "customer.npy"
        )

        # ====================================================
        # SIMILARITY MATRIX
        # ====================================================

        similarity = []

        for face in detected_faces:

            similarity.append(
                {
                    "face_id": face["id"],

                    "agent": self.cosine_similarity(
                        face["embedding"],
                        agent_embedding,
                    ),

                    "customer": self.cosine_similarity(
                        face["embedding"],
                        customer_embedding,
                    ),
                }
            )

        # ====================================================
        # FIND BEST ASSIGNMENT
        # ====================================================

        option1_score = (
            similarity[0]["agent"]
            + similarity[1]["customer"]
        )

        option2_score = (
            similarity[1]["agent"]
            + similarity[0]["customer"]
        )

        if option1_score >= option2_score:

            agent_face = similarity[0]
            customer_face = similarity[1]

        else:

            agent_face = similarity[1]
            customer_face = similarity[0]

        # ====================================================
        # PUBLIC MATRIX
        # ====================================================

        matrix = [
            {
                "face": row["face_id"],
                "agent_score": round(
                    row["agent"],
                    4,
                ),
                "customer_score": round(
                    row["customer"],
                    4,
                ),
            }
            for row in similarity
        ]

        # ====================================================
        # AGENT CHECK
        # ====================================================

        if (
            agent_face["agent"]
            < self.threshold
        ):

            return {
                "success": False,

                "message": (
                    "Agent verification failed."
                ),

                "agent_score": round(
                    agent_face["agent"],
                    4,
                ),

                "customer_score": round(
                    customer_face["customer"],
                    4,
                ),

                "matrix": matrix,
            }

        # ====================================================
        # CUSTOMER CHECK
        # ====================================================

        if (
            customer_face["customer"]
            < self.threshold
        ):

            return {
                "success": False,

                "message": (
                    "Customer verification failed."
                ),

                "agent_score": round(
                    agent_face["agent"],
                    4,
                ),

                "customer_score": round(
                    customer_face["customer"],
                    4,
                ),

                "matrix": matrix,
            }

        # ====================================================
        # SUCCESS
        # ====================================================

        return {
            "success": True,

            "agent": {
                "verified": True,

                "face_id": (
                    agent_face["face_id"]
                ),

                "score": round(
                    agent_face["agent"],
                    4,
                ),
            },

            "customer": {
                "verified": True,

                "face_id": (
                    customer_face["face_id"]
                ),

                "score": round(
                    customer_face["customer"],
                    4,
                ),
            },

            "matrix": matrix,
        }