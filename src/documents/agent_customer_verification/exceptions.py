from fastapi import HTTPException


class NoFaceDetectedException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=400,
            detail="No face detected."
        )


class MultipleFacesDetectedException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=400,
            detail="Multiple faces detected."
        )


class EmbeddingNotFoundException(HTTPException):
    def __init__(self, filename: str):
        super().__init__(
            status_code=404,
            detail=f"Embedding '{filename}' not found."
        )


class InvalidImageException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=400,
            detail="Unable to read image."
        )
