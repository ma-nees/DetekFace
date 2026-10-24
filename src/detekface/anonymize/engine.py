"""Face privacy and anonymization engine supporting blur, pixelation, and eye censor bars."""

import math
from typing import Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from detekface.types import BoundingBox, FaceDetection, FaceLandmarks, Point

SUPPORTED_ANONYMIZE_MODES = {"pixelate", "blur", "censor_eyes", "mosaic"}


def pixelate_roi(
    image: np.ndarray,
    bbox: Union[BoundingBox, Tuple[int, int, int, int]],
    block_size: int = 16,
) -> np.ndarray:
    """Apply an 8-bit retro mosaic pixelation to a bounding box region in-place.

    Args:
        image: BGR numpy image array to modify.
        bbox: Bounding box as BoundingBox instance or (x, y, w, h) tuple.
        block_size: Pixelation block size in pixels (default 16). Larger values = coarser blocks.

    Returns:
        The modified image.
    """
    if isinstance(bbox, BoundingBox):
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
    else:
        x, y, w, h = bbox[:4]

    img_h, img_w = image.shape[:2]
    x1 = max(0, min(x, img_w - 1))
    y1 = max(0, min(y, img_h - 1))
    x2 = max(0, min(x + w, img_w))
    y2 = max(0, min(y + h, img_h))

    roi_w = x2 - x1
    roi_h = y2 - y1
    if roi_w <= 2 or roi_h <= 2:
        return image

    bs = max(2, min(block_size, min(roi_w, roi_h)))
    small_w = max(1, roi_w // bs)
    small_h = max(1, roi_h // bs)

    roi = image[y1:y2, x1:x2]
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

    image[y1:y2, x1:x2] = pixelated
    return image


def blur_roi(
    image: np.ndarray,
    bbox: Union[BoundingBox, Tuple[int, int, int, int]],
    kernel_size: int = 51,
) -> np.ndarray:
    """Apply a heavy Gaussian frosted-glass blur to a bounding box region in-place.

    Args:
        image: BGR numpy image array to modify.
        bbox: Bounding box as BoundingBox instance or (x, y, w, h) tuple.
        kernel_size: Gaussian blur kernel size (must be an odd integer, default 51).

    Returns:
        The modified image.
    """
    if isinstance(bbox, BoundingBox):
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
    else:
        x, y, w, h = bbox[:4]

    img_h, img_w = image.shape[:2]
    x1 = max(0, min(x, img_w - 1))
    y1 = max(0, min(y, img_h - 1))
    x2 = max(0, min(x + w, img_w))
    y2 = max(0, min(y + h, img_h))

    roi_w = x2 - x1
    roi_h = y2 - y1
    if roi_w <= 2 or roi_h <= 2:
        return image

    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    k = max(3, min(k, min(roi_w, roi_h) | 1))

    roi = image[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (k, k), 0)

    image[y1:y2, x1:x2] = blurred
    return image


def censor_eyes_roi(
    image: np.ndarray,
    landmarks: FaceLandmarks,
    color: Tuple[int, int, int] = (0, 0, 0),
    thickness_factor: float = 0.35,
    padding_factor: float = 0.35,
) -> np.ndarray:
    """Draw a rotated solid redaction censor bar across eyes using facial landmarks.

    Args:
        image: BGR numpy image array to modify.
        landmarks: FaceLandmarks containing right_eye and left_eye coordinates.
        color: BGR bar color tuple (default is solid black (0, 0, 0)).
        thickness_factor: Bar thickness relative to interpupillary distance.
        padding_factor: Bar extension beyond eye centers.

    Returns:
        The modified image.
    """
    re = landmarks.right_eye
    le = landmarks.left_eye

    dx = le.x - re.x
    dy = le.y - re.y
    eye_dist = math.hypot(dx, dy)
    if eye_dist < 4:
        return image

    angle = math.atan2(dy, dx)
    cx = (re.x + le.x) / 2.0
    cy = (re.y + le.y) / 2.0

    bar_len = eye_dist * (1.0 + 2.0 * padding_factor)
    bar_thick = max(6.0, eye_dist * thickness_factor)

    rect = ((cx, cy), (bar_len, bar_thick), math.degrees(angle))
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.fillPoly(image, [box], color)
    return image


def anonymize(
    image: np.ndarray,
    faces: Sequence[FaceDetection],
    mode: str = "pixelate",
    block_size: int = 16,
    kernel_size: int = 51,
    copy: bool = True,
) -> np.ndarray:
    """Anonymize detected faces in an image using pixelation, blurring, or eye censor bars.

    Args:
        image: Source BGR image NumPy array.
        faces: Sequence of FaceDetection instances to anonymize.
        mode: Anonymization technique: 'pixelate' (or 'mosaic'), 'blur', 'censor_eyes'. Default is 'pixelate'.
        block_size: Block size for pixelate mode (default 16).
        kernel_size: Blur kernel size for blur mode (default 51).
        copy: Whether to copy the input image before modifying (default True).

    Returns:
        NumPy array with faces anonymized.
    """
    mode_clean = mode.lower().strip()
    if mode_clean not in SUPPORTED_ANONYMIZE_MODES:
        raise ValueError(
            f"Unsupported anonymization mode '{mode}'. Choose from: {sorted(SUPPORTED_ANONYMIZE_MODES)}"
        )

    out = image.copy() if copy else image

    for face in faces:
        if mode_clean in ("pixelate", "mosaic"):
            pixelate_roi(out, face.bbox, block_size=block_size)
        elif mode_clean == "blur":
            blur_roi(out, face.bbox, kernel_size=kernel_size)
        elif mode_clean == "censor_eyes":
            if face.landmarks is not None:
                censor_eyes_roi(out, face.landmarks)
            else:
                # Fallback to bounding box blur if landmarks are unavailable
                blur_roi(out, face.bbox, kernel_size=kernel_size)

    return out
