"""
==============================================================
ITR Detection Configuration
==============================================================

Central configuration for ITR Detection Engine.

This module acts as the single source of truth for
all detector settings.

No detector should contain hardcoded values.

==============================================================
"""

from dataclasses import dataclass

from .constants import (
    SUPPORTED_EXTENSIONS,
    MIN_DETECTION_CONFIDENCE,
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    LOW_CONFIDENCE,
    WEIGHTS,
    PRIMARY_KEYWORDS,
    SECONDARY_KEYWORDS,
    NEGATIVE_KEYWORDS,
    ITR_TYPES,
    MIN_PAGES,
    MAX_REASONABLE_PAGES,
    DIGITAL_TEXT_THRESHOLD,
    MIXED_TEXT_THRESHOLD,
    MAX_PAGES_TO_ANALYZE,
)


@dataclass(frozen=True)
class DetectionConfig:
    """
    Immutable Detection Configuration.
    """

    # ---------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------

    confidence_threshold: float = MIN_DETECTION_CONFIDENCE

    high_confidence: float = HIGH_CONFIDENCE

    medium_confidence: float = MEDIUM_CONFIDENCE

    low_confidence: float = LOW_CONFIDENCE


    # ---------------------------------------------------------
    # Supported Files
    # ---------------------------------------------------------

    supported_extensions = SUPPORTED_EXTENSIONS


    # ---------------------------------------------------------
    # Detection Weights
    # ---------------------------------------------------------

    metadata_weight: float = WEIGHTS["metadata"]

    keyword_weight: float = WEIGHTS["keyword"]

    layout_weight: float = WEIGHTS["layout"]

    structure_weight: float = WEIGHTS["structure"]


    # ---------------------------------------------------------
    # Keywords
    # ---------------------------------------------------------

    primary_keywords = PRIMARY_KEYWORDS

    secondary_keywords = SECONDARY_KEYWORDS

    negative_keywords = NEGATIVE_KEYWORDS


    # ---------------------------------------------------------
    # ITR Types
    # ---------------------------------------------------------

    itr_types = ITR_TYPES


    # ---------------------------------------------------------
    # Page Rules
    # ---------------------------------------------------------

    min_pages: int = MIN_PAGES

    max_pages: int = MAX_REASONABLE_PAGES


    # ---------------------------------------------------------
    # PDF Analysis
    # ---------------------------------------------------------

    digital_text_threshold: int = DIGITAL_TEXT_THRESHOLD

    mixed_text_threshold: int = MIXED_TEXT_THRESHOLD

    max_pages_to_analyze: int = MAX_PAGES_TO_ANALYZE


    # ---------------------------------------------------------
    # Future Flags
    # ---------------------------------------------------------

    enable_layout_detection: bool = True

    enable_keyword_detection: bool = True

    enable_metadata_detection: bool = True

    enable_structure_detection: bool = False

    enable_logo_detection: bool = False

    enable_signature_detection: bool = False


CONFIG = DetectionConfig()