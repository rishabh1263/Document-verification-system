"""
Agent Face Embedding Service.

Loads the registered Agent embedding and compares it against
a detected face embedding using cosine distance.
"""

from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

# Current file:
#
# src/
#   documents/
#     agent_property_verification/
#       services/
#         face/
#           embedding_service.py
#
# parents[2] = agent_property_verification
MODULE_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_PATH = (
    MODULE_ROOT
    / "embeddings"
    / "agent.npy"
)


# ============================================================
# EMBEDDING SERVICE
# ============================================================

class EmbeddingService:

    def __init__(
        self,
        threshold: float = 0.55,
        embedding_path: str | Path | None = None,
    ):

        self.threshold = threshold

        self.embedding_path = Path(
            embedding_path
            if embedding_path is not None
            else EMBEDDING_PATH
        ).resolve()

        self.agent_embedding = self._load_embedding()

    # ========================================================
    # LOAD REGISTERED AGENT EMBEDDING
    # ========================================================

    def _load_embedding(self) -> np.ndarray:

        if not self.embedding_path.is_file():
            raise FileNotFoundError(
                "Agent embedding not found: "
                f"{self.embedding_path}"
            )

        try:
            embedding = np.load(
                self.embedding_path,
                allow_pickle=False,
            )

        except Exception as exc:
            raise RuntimeError(
                "Unable to load Agent embedding from "
                f"{self.embedding_path}: {exc}"
            ) from exc

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).squeeze()

        if embedding.size == 0:
            raise ValueError(
                "Agent embedding file contains no data."
            )

        if embedding.ndim != 1:
            raise ValueError(
                "Agent embedding must be a 1-dimensional vector. "
                f"Received shape: {embedding.shape}"
            )

        if not np.all(np.isfinite(embedding)):
            raise ValueError(
                "Agent embedding contains invalid numeric values."
            )

        norm = float(
            np.linalg.norm(embedding)
        )

        if norm <= 0:
            raise ValueError(
                "Agent embedding has zero magnitude."
            )

        return embedding

    # ========================================================
    # COMPARE DETECTED FACE
    # ========================================================

    def compare(
        self,
        detected_embedding,
    ) -> dict:

        if detected_embedding is None:
            return {
                "verified": False,
                "similarity": 0.0,
            }

        try:

            registered = np.asarray(
                self.agent_embedding,
                dtype=np.float32,
            ).squeeze()

            detected = np.asarray(
                detected_embedding,
                dtype=np.float32,
            ).squeeze()

            if detected.ndim != 1:
                return {
                    "verified": False,
                    "similarity": 0.0,
                }

            if registered.shape != detected.shape:
                return {
                    "verified": False,
                    "similarity": 0.0,
                }

            if not np.all(
                np.isfinite(detected)
            ):
                return {
                    "verified": False,
                    "similarity": 0.0,
                }

            registered_norm = float(
                np.linalg.norm(registered)
            )

            detected_norm = float(
                np.linalg.norm(detected)
            )

            if (
                registered_norm <= 0
                or detected_norm <= 0
            ):
                return {
                    "verified": False,
                    "similarity": 0.0,
                }

            registered = (
                registered
                / registered_norm
            )

            detected = (
                detected
                / detected_norm
            )

            cosine_similarity = float(
                np.dot(
                    registered,
                    detected,
                )
            )

            cosine_similarity = float(
                np.clip(
                    cosine_similarity,
                    -1.0,
                    1.0,
                )
            )

            cosine_distance = (
                1.0
                - cosine_similarity
            )

            verified = bool(
                cosine_distance
                < self.threshold
            )

            return {
                "verified": verified,

                # Keep this key for compatibility with
                # the existing API.
                #
                # The old implementation returned distance
                # under the name "similarity".
                "similarity": round(
                    cosine_distance,
                    3,
                ),
            }

        except Exception:

            return {
                "verified": False,
                "similarity": 0.0,
            }


# ============================================================
# SHARED SERVICE INSTANCE
# ============================================================

embedding_service = EmbeddingService()