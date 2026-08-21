"""
src/utils.py

File-handling helpers for the Driving Licence Extractor.

Responsibilities:
1. Create required directories.
2. Find supported input documents.
3. Load normal image files with OpenCV.
4. Convert PDF pages to images using pdf2image + Poppler.
5. Return OpenCV images for the preprocessing/OCR pipeline.
"""

import os

import cv2
import numpy as np
from pdf2image import convert_from_path


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_IMAGE_EXT = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
)


# Poppler path
POPPLER_PATH = (
    r"C:\Users\Nidhi Singh\Downloads\Release-26.02.0-0"
    r"\poppler-26.02.0\Library\bin"
)


# ============================================================
# DIRECTORY HELPERS
# ============================================================

def ensure_dir(directory):
    """
    Create a directory if it does not already exist.

    Example:
        ensure_dir("output")
    """

    if not directory:
        return

    os.makedirs(
        directory,
        exist_ok=True
    )


# ============================================================
# INPUT FILE DISCOVERY
# ============================================================

def list_input_files(input_dir):
    """
    Return full paths of all supported files inside input_dir.

    Supported formats:
        JPG
        JPEG
        PNG
        BMP
        TIFF
        TIF
        PDF
    """

    # Automatically create input directory if missing
    ensure_dir(input_dir)

    files = []

    for fname in sorted(os.listdir(input_dir)):

        full_path = os.path.join(
            input_dir,
            fname
        )

        # Ignore folders
        if not os.path.isfile(full_path):
            continue

        ext = os.path.splitext(fname)[1].lower()

        if (
            ext in SUPPORTED_IMAGE_EXT
            or ext == ".pdf"
        ):
            files.append(full_path)

    return files


# ============================================================
# PDF LOADER
# ============================================================

def _load_pdf(file_path, pdf_dpi=300):
    """
    Convert every page of a PDF into an OpenCV image.

    Pipeline:

        PDF
         â†“
        pdf2image
         â†“
        Poppler
         â†“
        PIL Image
         â†“
        NumPy array
         â†“
        OpenCV BGR image
    """

    # --------------------------------------------------------
    # Verify Poppler installation
    # --------------------------------------------------------

    if not os.path.isdir(POPPLER_PATH):

        raise FileNotFoundError(
            "Poppler directory was not found.\n\n"
            f"Configured path:\n{POPPLER_PATH}"
        )

    pdftoppm_path = os.path.join(
        POPPLER_PATH,
        "pdftoppm.exe"
    )

    if not os.path.isfile(pdftoppm_path):

        raise FileNotFoundError(
            "pdftoppm.exe was not found.\n\n"
            f"Expected location:\n{pdftoppm_path}"
        )

    # --------------------------------------------------------
    # Convert PDF
    # --------------------------------------------------------

    try:

        pages = convert_from_path(
            file_path,
            dpi=pdf_dpi,
            poppler_path=POPPLER_PATH,
            fmt="png",
            thread_count=2,
        )

    except Exception as exc:

        raise RuntimeError(
            "\nPDF conversion failed.\n\n"
            f"File:\n{file_path}\n\n"
            f"Poppler:\n{POPPLER_PATH}\n\n"
            f"Original error:\n{exc}"
        ) from exc

    # --------------------------------------------------------
    # Check pages
    # --------------------------------------------------------

    if not pages:

        raise ValueError(
            f"No pages were extracted from PDF: {file_path}"
        )

    results = []

    # --------------------------------------------------------
    # PIL â†’ OpenCV
    # --------------------------------------------------------

    for page_number, page in enumerate(
        pages,
        start=1
    ):

        # Make sure page is RGB
        page = page.convert("RGB")

        # PIL â†’ NumPy
        np_image = np.array(page)

        # RGB â†’ BGR for OpenCV
        image = cv2.cvtColor(
            np_image,
            cv2.COLOR_RGB2BGR
        )

        label = (
            f"{os.path.basename(file_path)}"
            f"_page{page_number}"
        )

        results.append(
            (
                label,
                image
            )
        )

    return results


# ============================================================
# IMAGE LOADER
# ============================================================

def _load_image(file_path):
    """
    Load a normal image using OpenCV.

    Returns:
        [
            (filename, image)
        ]
    """

    image = cv2.imread(
        file_path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            f"Could not read image file: {file_path}"
        )

    label = os.path.basename(
        file_path
    )

    return [
        (
            label,
            image
        )
    ]


# ============================================================
# MAIN DOCUMENT LOADER
# ============================================================

def load_file_as_images(
    file_path,
    pdf_dpi=300
):
    """
    Load either an image or PDF.

    Returns a list of:

        [
            (page_label, opencv_image),
            ...
        ]

    Example image:

        licence.jpg

    returns:

        [
            ("licence.jpg", image)
        ]


    Example PDF:

        licence.pdf

    returns:

        [
            ("licence.pdf_page1", image),
            ("licence.pdf_page2", image)
        ]
    """

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not os.path.isfile(file_path):

        raise FileNotFoundError(
            f"Input file does not exist: {file_path}"
        )

    # --------------------------------------------------------
    # Determine extension
    # --------------------------------------------------------

    ext = os.path.splitext(
        file_path
    )[1].lower()

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if ext == ".pdf":

        return _load_pdf(
            file_path=file_path,
            pdf_dpi=pdf_dpi
        )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if ext in SUPPORTED_IMAGE_EXT:

        return _load_image(
            file_path=file_path
        )

    # --------------------------------------------------------
    # Unsupported format
    # --------------------------------------------------------

    raise ValueError(
        f"Unsupported file type: {ext}"
    )
