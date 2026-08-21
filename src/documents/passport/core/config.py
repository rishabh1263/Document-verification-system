"""
Passport module configuration.

IMPORTANT:
- Keep passport-specific configuration here.
- Keep generic document/security/validation logic inside src.common.
- Configuration should support the LOS document-verification flow.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PASSPORT_ROOT = Path(__file__).resolve().parents[1]

PASSPORT_STORAGE_DIR = (
    PASSPORT_ROOT / "storage_data"
)


# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Passport-specific operational configuration.

    Generic validation/security rules should NOT be duplicated here.
    """

    # -----------------------------------------------------------------------
    # APPLICATION
    # -----------------------------------------------------------------------

    APP_NAME: str = "Passport Verification"

    APP_VERSION: str = "2.0.0"

    DEBUG: bool = False


    # -----------------------------------------------------------------------
    # SERVER
    # -----------------------------------------------------------------------

    HOST: str = "127.0.0.1"

    PORT: int = 8000


    # -----------------------------------------------------------------------
    # STORAGE
    # -----------------------------------------------------------------------

    STORAGE_DIR: str = str(
        PASSPORT_STORAGE_DIR
    )

    UPLOAD_DIR: str = str(
        PASSPORT_STORAGE_DIR / "uploads"
    )

    LOG_DIR: str = str(
        PASSPORT_STORAGE_DIR / "logs"
    )

    REPORT_DIR: str = str(
        PASSPORT_STORAGE_DIR / "reports"
    )

    METADATA_DIR: str = str(
        PASSPORT_STORAGE_DIR / "metadata"
    )


    # -----------------------------------------------------------------------
    # PDF
    # -----------------------------------------------------------------------

    # Passport normally requires only the identity/details page.
    # Allowing 2 pages gives us room for a second page without scanning
    # an unnecessarily large document.
    MAX_PDF_PAGES: int = 2


    # -----------------------------------------------------------------------
    # LOS LATENCY
    # -----------------------------------------------------------------------

    # Target for synchronous document validation.
    #
    # This is a target, not a guarantee.
    # We should monitor P95 latency in production rather than assuming
    # every request will always complete below this number.
    MAX_PROCESSING_MS: int = 1000


    # -----------------------------------------------------------------------
    # OCR
    # -----------------------------------------------------------------------

    # Avoid processing unnecessarily huge images.
    OCR_MAX_WIDTH: int = 1400

    # PP-OCRv5 mobile detector limit used in the fast LOS path.
    OCR_DET_LIMIT_SIDE_LEN: int = 640


    # -----------------------------------------------------------------------
    # PASSPORT VALIDATION
    # -----------------------------------------------------------------------

    # Minimum common image-quality score required before OCR.
    MIN_QUALITY_SCORE: float = 50.0

    # Minimum OCR confidence considered useful evidence.
    MIN_OCR_CONFIDENCE: float = 0.60

    # Minimum score for automatic local document pass.
    PASS_SCORE: float = 80.0

    # Score below this is normally rejected by the common validation layer.
    REVIEW_SCORE: float = 60.0


    # -----------------------------------------------------------------------
    # MRZ
    # -----------------------------------------------------------------------

    # TD3 passport MRZ contains exactly two lines.
    MRZ_LINE_COUNT: int = 2

    # Each TD3 MRZ line contains 44 characters.
    MRZ_LINE_LENGTH: int = 44


    # -----------------------------------------------------------------------
    # LOS / RCU BEHAVIOUR
    # -----------------------------------------------------------------------

    # The synchronous LOS path should NOT run expensive deep tamper analysis.
    #
    # If the document reaches MANUAL_REVIEW or DOCUMENT_SUSPICIOUS,
    # RCU can perform deeper analysis separately.
    ENABLE_FAST_PATH_TAMPER: bool = False

    ENABLE_RCU_STAGE: bool = True


    # -----------------------------------------------------------------------
    # AUTHORITATIVE VERIFICATION
    # -----------------------------------------------------------------------

    # Local document validation must never pretend to be government/source-
    # of-truth verification.
    AUTHORITATIVE_VERIFICATION_ENABLED: bool = False


    # -----------------------------------------------------------------------
    # STORAGE / CLEANUP
    # -----------------------------------------------------------------------

    # Keep uploaded files only when required by the LOS/RCU workflow.
    STORE_UPLOADS: bool = True

    STORE_METADATA: bool = True


    # -----------------------------------------------------------------------
    # PYDANTIC SETTINGS
    # -----------------------------------------------------------------------

    model_config = SettingsConfigDict(
        extra="ignore"
    )


# ---------------------------------------------------------------------------
# SINGLE SETTINGS INSTANCE
# ---------------------------------------------------------------------------

settings = Settings()