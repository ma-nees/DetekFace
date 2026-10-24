"""Unit tests for Terminal ASCII face renderer."""

import numpy as np
import pytest

from detekface.ascii import ASCII_RAMP_SHORT, frame_to_ascii
from detekface.types import BoundingBox, FaceDetection


def test_frame_to_ascii_grayscale():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Bright center square
    img[30:70, 30:70] = 255

    ascii_text = frame_to_ascii(img, cols=40, rows=20, mode="gray")
    assert isinstance(ascii_text, str)
    lines = ascii_text.split("\n")
    assert len(lines) == 20
    assert len(lines[0]) == 40
    # Center lines should have bright character '@'
    assert "@" in ascii_text


def test_frame_to_ascii_color_and_matrix():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[10:40, 10:40] = (0, 200, 0)

    color_out = frame_to_ascii(img, cols=30, rows=15, mode="color")
    assert "\033[38;2;" in color_out

    matrix_out = frame_to_ascii(img, cols=30, rows=15, mode="matrix")
    assert "\033[38;2;0;" in matrix_out


def test_frame_to_ascii_with_faces():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    face = FaceDetection(bbox=BoundingBox(20, 20, 40, 40), confidence=0.92)

    ascii_text = frame_to_ascii(img, faces=[face], cols=40, rows=20, mode="gray")
    # Face reticle border characters should be present
    assert "+" in ascii_text or "|" in ascii_text or "-" in ascii_text
