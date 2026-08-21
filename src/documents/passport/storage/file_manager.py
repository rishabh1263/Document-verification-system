from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import UploadFile

from src.documents.passport.core.config import settings


Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


class FileManager:

    @staticmethod
    def save(file: UploadFile):

        extension = Path(file.filename).suffix

        filename = f"{uuid4()}{extension}"

        filepath = Path(settings.UPLOAD_DIR) / filename

        with filepath.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "filename": filename,
            "path": str(filepath)
        }
