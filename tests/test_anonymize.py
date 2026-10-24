"""Unit tests for Face Anonymization and Privacy engine."""

from pathlib import Path
import cv2
import numpy as np
import pytest

import detekface
from detekface.anonymize import (
    SUPPORTED_ANONYMIZE_MODES,
    anonymize,
    blur_roi,
    censor_eyes_roi,
    pixelate_roi,
)
from detekface.types import BoundingBox, FaceDetection, FaceLandmarks, Point

ASSETS_DIR = Path(__file__).parent / "assets"
SAMPLE_FACE = ASSETS_DIR / "sample_face.jpg"


def test_supported_modes():
    assert "pixelate" in SUPPORTED_ANONYMIZE_MODES
    assert "blur" in SUPPORTED_ANONYMIZE_MODES
    assert "censor_eyes" in SUPPORTED_ANONYMIZE_MODES


def test_pixelate_roi():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)
    original = img.copy()

    bbox = BoundingBox(50, 50, 100, 100)
    out = pixelate_roi(img, bbox, block_size=10)

    assert out.shape == original.shape
    assert not np.array_equal(out, original)
    # Outside ROI should remain untouched
    assert np.array_equal(out[:40, :40], original[:40, :40])


def test_blur_roi():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (140, 140), (200, 200, 200), -1)
    original = img.copy()

    bbox = BoundingBox(50, 50, 100, 100)
    out = blur_roi(img, bbox, kernel_size=25)

    assert out.shape == original.shape
    assert not np.array_equal(out, original)
    # Outside ROI remains unchanged
    assert np.array_equal(out[:40, :40], original[:40, :40])


def test_censor_eyes_roi():
    img = np.ones((200, 200, 3), dtype=np.uint8) * 200
    original = img.copy()

    landmarks = FaceLandmarks(
        right_eye=Point(70, 90),
        left_eye=Point(130, 90),
        nose_tip=Point(100, 120),
        right_mouth=Point(75, 150),
        left_mouth=Point(125, 150),
    )

    out = censor_eyes_roi(img, landmarks)
    assert not np.array_equal(out, original)
    # Eye centers should now be black (0, 0, 0)
    assert np.all(out[90, 70] == 0)
    assert np.all(out[90, 130] == 0)


def test_anonymize_modes():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (90, 90), 40, (255, 255, 255), -1)
    face = FaceDetection(
        bbox=BoundingBox(50, 50, 80, 80),
        confidence=0.95,
        landmarks=FaceLandmarks(
            right_eye=Point(70, 75),
            left_eye=Point(110, 75),
            nose_tip=Point(90, 95),
            right_mouth=Point(75, 115),
            left_mouth=Point(105, 115),
        ),
    )

    for mode in ["pixelate", "blur", "censor_eyes", "mosaic"]:
        out = anonymize(img, [face], mode=mode, copy=True)
        assert out.shape == img.shape
        assert not np.array_equal(out, img)



def test_anonymize_invalid_mode():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported anonymization mode"):
        anonymize(img, [], mode="unknown_mode")


def test_detection_result_anonymize_integration():
    if not SAMPLE_FACE.exists():
        pytest.skip(f"Asset {SAMPLE_FACE} missing.")

    result = detekface.detect(SAMPLE_FACE)
    assert result.has_faces

    # Pixelate mode
    pixelated = result.anonymize(mode="pixelate", block_size=14)
    assert pixelated.shape == result.image.shape
    assert not np.array_equal(pixelated, result.image)

    # Blur mode
    blurred = result.anonymize(mode="blur")
    assert blurred.shape == result.image.shape
    assert not np.array_equal(blurred, result.image)

    # Censor eyes mode
    censored = result.anonymize(mode="censor_eyes")
    assert censored.shape == result.image.shape
    assert not np.array_equal(censored, result.image)
