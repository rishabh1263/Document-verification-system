"""
Base Extractor

All document extractors (Sale Deed, Aadhaar, PAN, etc.)
must inherit from this class.
"""

from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    """
    Abstract base class for all document extractors.
    """

    @abstractmethod
    def extract(self, text: str) -> dict:
        """
        Extract structured information from OCR text.

        Parameters
        ----------
        text : str
            OCR extracted text.

        Returns
        -------
        dict
            Dictionary containing extracted fields.
        """
        pass
