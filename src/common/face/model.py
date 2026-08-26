from functools import lru_cache

from insightface.app import FaceAnalysis


@lru_cache(maxsize=1)
def get_face_app() -> FaceAnalysis:
    """
    Return the single shared InsightFace buffalo_l instance.

    The model is initialized only once per process.
    """

    app = FaceAnalysis(
        name="buffalo_l",
        providers=[
            "CPUExecutionProvider",
        ],
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640),
    )

    return app