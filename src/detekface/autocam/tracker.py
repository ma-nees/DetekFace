"""Center Stage AI Cameraman: smooth face-tracking auto-zoom and framing engine."""

from typing import List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from detekface.types import BoundingBox, FaceDetection


class CenterStageTracker:
    """Smoothly tracks and frames detected faces with dynamic auto-zoom and cinematic panning."""

    def __init__(
        self,
        smoothing: float = 0.08,
        headroom_ratio: float = 0.45,
        margin_scale: float = 2.4,
        max_zoom: float = 2.8,
        show_pip: bool = True,
    ) -> None:
        """Initialize CenterStageTracker.

        Args:
            smoothing: Exponential smoothing factor (0.01 = very slow/cinematic, 0.2 = fast). Default 0.08.
            headroom_ratio: Headroom proportion above the face top (default 0.45).
            margin_scale: Face size multiplier for bounding crop framing (default 2.4).
            max_zoom: Maximum zoom magnification factor (default 2.8x).
            show_pip: Whether to overlay a miniature Picture-in-Picture wide-angle view. Default True.
        """
        self.smoothing = max(0.01, min(0.5, float(smoothing)))
        self.headroom_ratio = float(headroom_ratio)
        self.margin_scale = float(margin_scale)
        self.max_zoom = max(1.0, float(max_zoom))
        self.show_pip = bool(show_pip)

        # Smooth state tracking (x, y, w, h)
        self._current_crop: Optional[Tuple[float, float, float, float]] = None
        self._target_crop: Optional[Tuple[float, float, float, float]] = None

    def reset(self) -> None:
        """Reset internal smoothing filters to default wide-angle state."""
        self._current_crop = None
        self._target_crop = None

    def _calculate_target_box(
        self,
        frame_shape: Tuple[int, int],
        faces: Sequence[FaceDetection],
    ) -> Tuple[float, float, float, float]:
        """Calculate the ideal target framing box encompassing the active faces."""
        h, w = frame_shape[:2]
        frame_aspect = w / h

        if not faces:
            # When no face is detected, gently pull back to the full frame view
            return (0.0, 0.0, float(w), float(h))

        # Find enclosing bounding box for all detected faces
        min_x = min(f.bbox.x for f in faces)
        min_y = min(f.bbox.y for f in faces)
        max_x = max(f.bbox.x2 for f in faces)
        max_y = max(f.bbox.y2 for f in faces)

        face_w = max_x - min_x
        face_h = max_y - min_y
        center_x = (min_x + max_x) / 2.0
        # Position center slightly lower than the face midpoint to keep eyes in upper third
        center_y = min_y + face_h * (0.5 + self.headroom_ratio * 0.4)

        # Calculate target crop dimensions matching the original aspect ratio
        raw_target_h = max(face_h * self.margin_scale, (face_w * self.margin_scale) / frame_aspect)
        min_allowed_h = h / self.max_zoom
        target_h = max(min_allowed_h, min(float(h), raw_target_h))
        target_w = target_h * frame_aspect

        # Clamp center so crop box stays inside frame boundaries
        half_w = target_w / 2.0
        half_h = target_h / 2.0

        clamped_center_x = max(half_w, min(w - half_w, center_x))
        clamped_center_y = max(half_h, min(h - half_h, center_y))

        crop_x = clamped_center_x - half_w
        crop_y = clamped_center_y - half_h

        return (crop_x, crop_y, target_w, target_h)

    def process_frame(
        self,
        frame: np.ndarray,
        faces: Sequence[FaceDetection],
    ) -> Tuple[np.ndarray, float]:
        """Apply smooth auto-zoom framing to the input frame.

        Args:
            frame: Input BGR numpy image array.
            faces: Sequence of detected FaceDetection instances.

        Returns:
            Tuple of (framed_frame, current_zoom_level).
        """
        h, w = frame.shape[:2]
        target = self._calculate_target_box((h, w), faces)

        # Initialize current crop on first frame
        if self._current_crop is None:
            self._current_crop = target
        else:
            # Exponential moving average filter for buttery smooth cinematic motion
            cx, cy, cw, ch = self._current_crop
            tx, ty, tw, th = target
            alpha = self.smoothing

            new_x = cx + alpha * (tx - cx)
            new_y = cy + alpha * (ty - cy)
            new_w = cw + alpha * (tw - cw)
            new_h = ch + alpha * (th - ch)
            self._current_crop = (new_x, new_y, new_w, new_h)

        crop_x, crop_y, crop_w, crop_h = self._current_crop

        # Ensure valid integer crop indices
        ix1 = max(0, min(int(round(crop_x)), w - 2))
        iy1 = max(0, min(int(round(crop_y)), h - 2))
        ix2 = max(ix1 + 2, min(int(round(crop_x + crop_w)), w))
        iy2 = max(iy1 + 2, min(int(round(crop_y + crop_h)), h))

        cropped_roi = frame[iy1:iy2, ix1:ix2]
        if cropped_roi.size == 0:
            return frame.copy(), 1.0

        # High quality upscale back to original camera resolution
        framed = cv2.resize(cropped_roi, (w, h), interpolation=cv2.INTER_LINEAR)
        zoom_level = h / max(1.0, float(iy2 - iy1))

        # Render optional Picture-in-Picture mini radar in the top-right corner
        if self.show_pip:
            pip_w = max(120, int(w * 0.22))
            pip_h = int(pip_w / (w / h))
            pip_thumb = cv2.resize(frame, (pip_w, pip_h), interpolation=cv2.INTER_AREA)

            # Draw green tracking reticle on the PIP thumbnail
            scale_x = pip_w / w
            scale_y = pip_h / h
            rx1 = int(round(ix1 * scale_x))
            ry1 = int(round(iy1 * scale_y))
            rx2 = int(round(ix2 * scale_x))
            ry2 = int(round(iy2 * scale_y))

            cv2.rectangle(pip_thumb, (rx1, ry1), (rx2, ry2), (0, 255, 136), 1)

            # Insert PIP thumbnail into top-right with margin and border
            margin = 15
            px1 = w - pip_w - margin
            py1 = margin
            px2 = px1 + pip_w
            py2 = py1 + pip_h

            framed[py1:py2, px1:px2] = pip_thumb
            cv2.rectangle(framed, (px1 - 1, py1 - 1), (px2 + 1, py2 + 1), (50, 50, 50), 1)
            cv2.putText(
                framed,
                "WIDE RADAR",
                (px1 + 4, py2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 136),
                1,
                cv2.LINE_AA,
            )

        return framed, zoom_level
