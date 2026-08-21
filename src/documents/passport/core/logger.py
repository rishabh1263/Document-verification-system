import logging
import os

from src.documents.passport.core.config import settings


os.makedirs(settings.LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(settings.LOG_DIR, "application.log")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("passport_verification")
