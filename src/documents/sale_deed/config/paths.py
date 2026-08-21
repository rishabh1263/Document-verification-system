"""
Centralized file and folder paths for the Sale Deed AI project.
"""

from pathlib import Path

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# DATA FOLDERS
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"

INPUT_DIR = DATA_DIR / "input"
INPUT_PDF_DIR = INPUT_DIR / "pdf"
INPUT_IMAGE_DIR = INPUT_DIR / "images"

PROCESSED_DIR = DATA_DIR / "processed"
ENHANCED_DIR = PROCESSED_DIR / "enhanced"
QUALITY_REPORT_DIR = PROCESSED_DIR / "quality_reports"

OUTPUT_DIR = DATA_DIR / "output"
JSON_OUTPUT_DIR = OUTPUT_DIR / "json"
TABLE_OUTPUT_DIR = OUTPUT_DIR / "tables"
OCR_OUTPUT_DIR = OUTPUT_DIR / "ocr"
LOG_OUTPUT_DIR = OUTPUT_DIR / "logs"

# ==========================================================
# OTHER PROJECT FOLDERS
# ==========================================================

MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
TESTS_DIR = PROJECT_ROOT / "tests"
