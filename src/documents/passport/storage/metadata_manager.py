import json
from pathlib import Path
from datetime import datetime

from src.documents.passport.core.config import settings


class MetadataManager:

    @staticmethod
    def save(request_id: str, data: dict):

        Path(settings.METADATA_DIR).mkdir(parents=True, exist_ok=True)

        file = Path(settings.METADATA_DIR) / f"{request_id}.json"

        data["created_at"] = datetime.utcnow().isoformat()

        with open(file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
