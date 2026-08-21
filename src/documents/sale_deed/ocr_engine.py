"""
High Performance OCR Engine
Batch OCR using PaddleOCR 3.7 â€” Fixed text extraction
"""

import time
from pathlib import Path

import cv2
from paddleocr import PaddleOCR


class OCREngine:

    def __init__(self):
        print("=" * 70)
        print("Loading PaddleOCR...")
        print("=" * 70)

        start = time.time()

        self.ocr = PaddleOCR(
            lang="hi",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=True,
            cpu_threads=8
        )

        print(f"PaddleOCR Loaded in {time.time()-start:.2f} sec")
        print("=" * 70)

        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------

    def _resize_image(self, image):
        h, w = image.shape[:2]
        max_side = 1600
        if max(h, w) <= max_side:
            return image
        scale = max_side / max(h, w)
        return cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    # ---------------------------------------------------------

    def _is_blank_page(self, image):
        """Check if a page is effectively blank."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.countNonZero(255 - gray) < 1000

    # ---------------------------------------------------------

    def _extract_result_text(self, result):
        """
        Extract text lines from PaddleOCR result.

        PaddleOCR returns: list of [bbox, (text, confidence)]
        Example: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ('text', 0.98)]
        """
        lines = []

        if result is None:
            return ""

        try:
            # PaddleOCR returns list of detection results
            if isinstance(result, list):
                for line_data in result:
                    if line_data is None:
                        continue
                    # Each item: [bbox_coords, (text, confidence)]
                    if isinstance(line_data, list) and len(line_data) >= 2:
                        text_info = line_data[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                            text = str(text_info[0]).strip()
                            if text:
                                lines.append(text)
                        elif isinstance(text_info, str):
                            text = text_info.strip()
                            if text:
                                lines.append(text)

            # Fallback: try dict format (some versions)
            elif isinstance(result, dict):
                texts = result.get("rec_texts", [])
                for text in texts:
                    text = str(text).strip()
                    if text:
                        lines.append(text)

        except Exception as e:
            print(f"Page parsing failed : {e}")

        return "\n".join(lines)

    # =========================================================
    # Step 1 â€“ Preview OCR (Page 1, Page 2, Last Page)
    # =========================================================

    def preview_ocr(self, images):
        """OCR only Page 1, Page 2 and Last Page."""
        total = len(images)
        indices = [0]
        if total > 1:
            indices.append(1)
        if total > 2:
            indices.append(total - 1)

        preview_images = [self._resize_image(images[i]) for i in indices]

        print()
        print("=" * 70)
        print(f"Running Preview OCR ({len(preview_images)} pages)")
        print("=" * 70)

        results = self.ocr.predict(preview_images)

        text = []
        for idx, result in enumerate(results):
            page_no = indices[idx] + 1
            page_text = self._extract_result_text(result)
            print(f"âœ“ Page {page_no} | {len(page_text)} chars")
            text.append(f"\n========== PAGE {page_no} ==========\n" + page_text)

        return "\n".join(text)

    # =========================================================
    # Step 2 â€“ Remaining OCR (all except Page 1, 2, Last)
    # =========================================================

    def remaining_ocr(self, images):
        """OCR all pages except Page1, Page2 and Last Page."""
        total = len(images)
        skip = {0, 1, total - 1}
        indices = [i for i in range(total) if i not in skip]

        if not indices:
            return ""

        remaining_images = [self._resize_image(images[i]) for i in indices]

        print()
        print("=" * 70)
        print(f"Running Remaining OCR ({len(indices)} pages)")
        print("=" * 70)

        results = self.ocr.predict(remaining_images)

        text = []
        for idx, result in enumerate(results):
            page_no = indices[idx] + 1
            page_text = self._extract_result_text(result)
            print(f"âœ“ Page {page_no} | {len(page_text)} chars")
            text.append(f"\n========== PAGE {page_no} ==========\n" + page_text)

        return "\n".join(text)

    # =========================================================
    # Step 3 â€“ Full OCR (all pages at once)
    # =========================================================

    def full_ocr(self, images):
        processed = [self._resize_image(img) for img in images]
        results = self.ocr.predict(processed)

        text = []
        for page in results:
            text.append(self._extract_result_text(page))

        return "\n".join(text)

    # =========================================================
    # Main Method â€“ Full OCR from file paths
    # =========================================================

    def extract_text(self, image_paths):
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]

        start = time.time()

        print("=" * 70)
        print("Reading Images...")
        print("=" * 70)

        images = []
        page_numbers = []

        for idx, path in enumerate(image_paths, start=1):
            img = cv2.imread(str(path))
            if img is None:
                print(f"Cannot read {path}")
                continue

            if self._is_blank_page(img):
                print(f"Blank page skipped : {path}")
                continue

            img = self._resize_image(img)
            images.append(img)
            page_numbers.append(idx)

        if not images:
            return ""

        # Run OCR with single instance (PaddleOCR handles batch internally)
        print()
        print("=" * 70)
        print(f"Running OCR ({len(images)} pages)")
        print("=" * 70)

        results = self.ocr.predict(images)

        print("OCR completed.\n")

        final_text = []
        total_chars = 0

        for page_no, result in zip(page_numbers, results):
            page_text = self._extract_result_text(result)
            final_text.append(f"\n========== PAGE {page_no} ==========\n")
            final_text.append(page_text)
            total_chars += len(page_text)
            print(f"âœ“ Page {page_no} : {len(page_text.splitlines())} lines | {len(page_text)} chars")

        final_text = "\n".join(final_text)

        output_file = self.output_dir / "ocr_output.txt"
        output_file.write_text(final_text, encoding="utf-8")

        print()
        print("=" * 70)
        print("OCR SUMMARY")
        print("=" * 70)
        print(f"Pages      : {len(images)}")
        print(f"Characters : {total_chars}")
        print(f"Saved      : {output_file}")
        print(f"Time       : {time.time()-start:.2f} sec")
        print("=" * 70)

        return final_text
