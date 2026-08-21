"""
==============================================================
ITR Detection Utilities
==============================================================

Reusable helper functions.

No business logic should exist here.

Author : SBFC Document Intelligence
==============================================================
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any


# ==========================================================
# FILE HELPERS
# ==========================================================

def file_exists(file_path: str) -> bool:
    """
    Check if file exists.
    """
    return Path(file_path).exists()


def get_extension(file_path: str) -> str:
    """
    Return lowercase file extension.
    """
    return Path(file_path).suffix.lower()


def get_filename(file_path: str) -> str:
    """
    Return filename only.
    """
    return Path(file_path).name


def get_file_size(file_path: str) -> int:
    """
    Return file size in bytes.
    """
    return os.path.getsize(file_path)


# ==========================================================
# HASHING
# ==========================================================

def generate_sha256(file_path: str) -> str:
    """
    Generate SHA256 hash.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            chunk = f.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ==========================================================
# TEXT
# ==========================================================

def normalize_text(text: str) -> str:
    """
    Normalize extracted text.
    """

    if not text:
        return ""

    text = text.lower()

    text = text.replace("\n", " ")

    text = text.replace("\t", " ")

    text = " ".join(text.split())

    return text


# ==========================================================
# NUMERIC
# ==========================================================

def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0
) -> float:
    """
    Clamp value.
    """

    return max(minimum, min(maximum, value))


def safe_float(
    value: Any,
    default: float = 0.0
) -> float:
    """
    Safe float conversion.
    """

    try:
        return float(value)

    except Exception:
        return default


# ==========================================================
# TIMER
# ==========================================================

def current_time() -> float:
    """
    High precision timer.
    """

    return time.perf_counter()


def elapsed_time(start: float) -> float:
    """
    Return elapsed milliseconds.
    """

    return round(
        (time.perf_counter() - start) * 1000,
        2
    )