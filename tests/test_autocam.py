"""Unit tests for Center Stage auto-framing tracker."""

import numpy as np
import pytest

from detekface.autocam import CenterStageTracker
from detekface.types import BoundingBox, FaceDetection


def test_center_stage_init():
    tracker = CenterStageTracker(smoothing=0.1, max_zoom=3.0, show_pip=True)
    assert tracker.smoothing == 0.1
    assert tracker.max_zoom == 3.0
    assert tracker.show_pip is True


def test_center_stage_no_face_fallback():
    tracker = CenterStageTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    framed, zoom = tracker.process_frame(frame, [])
    assert framed.shape == frame.shape
    assert zoom >= 1.0


def test_center_stage_tracking_smooth_zoom():
    tracker = CenterStageTracker(smoothing=0.2, max_zoom=2.5, show_pip=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Place a simulated face centered at (300, 200) with size 80x80
    face = FaceDetection(
        bbox=BoundingBox(x=260, y=160, width=80, height=80),
        confidence=0.9,
    )

    framed1, zoom1 = tracker.process_frame(frame, [face])
    assert framed1.shape == frame.shape
    assert zoom1 >= 1.0

    # Process another frame to verify smoothing motion
    framed2, zoom2 = tracker.process_frame(frame, [face])
    assert framed2.shape == frame.shape
    assert zoom2 >= zoom1  # Zoom should progressively zoom in


def test_center_stage_reset():
    tracker = CenterStageTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    face = FaceDetection(bbox=BoundingBox(200, 200, 100, 100), confidence=0.9)

    tracker.process_frame(frame, [face])
    assert tracker._current_crop is not None

    tracker.reset()
    assert tracker._current_crop is None
