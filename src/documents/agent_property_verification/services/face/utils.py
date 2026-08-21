import cv2
import numpy as np
import torch


def xyxy2xywh(bbox):

    if isinstance(bbox, list):
        bbox = np.array(bbox)

    result = np.copy(bbox)

    result[..., 2] = bbox[..., 2] - bbox[..., 0]
    result[..., 3] = bbox[..., 3] - bbox[..., 1]

    return result


def crop_face(
    image,
    bbox,
    scale,
    out_w,
    out_h,
):

    src_h, src_w = image.shape[:2]

    x, y, box_w, box_h = bbox

    scale = min(
        (src_h - 1) / box_h,
        (src_w - 1) / box_w,
        scale
    )

    new_w = box_w * scale
    new_h = box_h * scale

    center_x = x + box_w / 2
    center_y = y + box_h / 2

    x1 = max(0, int(center_x - new_w / 2))
    y1 = max(0, int(center_y - new_h / 2))

    x2 = min(src_w - 1, int(center_x + new_w / 2))
    y2 = min(src_h - 1, int(center_y + new_h / 2))

    cropped = image[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    return cv2.resize(
        cropped,
        (out_w, out_h)
    )


def to_tensor(image):

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    return torch.from_numpy(
        image.transpose(2, 0, 1)
    ).float()


def draw_bbox(
    image,
    bbox,
    label,
    score,
    color=None,
):

    x, y, w, h = bbox

    if color is None:
        color = (
            (0, 255, 0)
            if label == "Real"
            else (0, 0, 255)
        )

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        color,
        2
    )

    cv2.putText(
        image,
        f"{label}: {score:.2f}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    return image
