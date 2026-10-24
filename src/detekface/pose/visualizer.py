"""Visualizer for rendering 3D head pose orientation axes and attention status HUD."""

from typing import Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from detekface.pose.estimator import HeadPose, HeadPoseEstimator, _default_estimator
from detekface.types import BoundingBox, FaceDetection, FaceLandmarks


def draw_head_pose(
    image: np.ndarray,
    pose: HeadPose,
    bbox: Optional[Union[BoundingBox, Tuple[int, int, int, int]]] = None,
    show_axes: bool = True,
    show_badge: bool = True,
    axis_length: float = 60.0,
    axis_thickness: int = 2,
    estimator: Optional[HeadPoseEstimator] = None,
) -> np.ndarray:
    """Draw 3D head pose orientation axes and an attention state badge onto an image.

    Args:
        image: BGR numpy image array.
        pose: HeadPose instance containing estimated orientation.
        bbox: Optional face bounding box used to position the HUD badge.
        show_axes: Whether to render the 3D RGB orientation arrows.
        show_badge: Whether to render the HUD attention status badge.
        axis_length: Length of the projected 3D axis arrows in mm.
        axis_thickness: Pixel thickness of the projected axis lines.
        estimator: Optional HeadPoseEstimator instance for projecting axes.

    Returns:
        The annotated image (modified in-place if contiguous, or a copy).
    """
    canvas = image.copy()
    h, w = canvas.shape[:2]
    est = estimator or _default_estimator

    # 1. Draw 3D Orientation Axes
    if show_axes and pose.rvec is not None and pose.tvec is not None:
        axes = est.project_axes(pose, (h, w), axis_length=axis_length)
        if axes is not None:
            origin = axes["origin"]
            x_pt = axes["x_axis"]
            y_pt = axes["y_axis"]
            z_pt = axes["z_axis"]

            # X-axis: Red (Viewer Right / Subject Left) - BGR: (0, 0, 255)
            cv2.arrowedLine(canvas, origin, x_pt, (0, 0, 255), axis_thickness, tipLength=0.2)
            # Y-axis: Green (Down) - BGR: (0, 255, 0)
            cv2.arrowedLine(canvas, origin, y_pt, (0, 255, 0), axis_thickness, tipLength=0.2)
            # Z-axis: Blue (Gaze vector forward) - BGR: (255, 128, 0) or vibrant cyan/blue (255, 200, 0)
            cv2.arrowedLine(canvas, origin, z_pt, (255, 200, 0), max(axis_thickness, 3), tipLength=0.25)

            # Draw a glowing dot at the nose origin
            cv2.circle(canvas, origin, 3, (255, 255, 255), -1)

    # 2. Draw Attention & Gaze HUD Badge
    if show_badge:
        # Determine badge position
        if bbox is not None:
            if isinstance(bbox, BoundingBox):
                bx, by, bw, bh = bbox.x, bbox.y, bbox.width, bbox.height
            else:
                bx, by, bw, bh = bbox[:4]
            badge_x = max(10, bx)
            badge_y = max(30, by - 12)
        else:
            badge_x = 16
            badge_y = 36

        # Status text & colors
        if pose.is_looking_at_camera:
            status_text = f"ATTENTIVE ({pose.direction})"
            badge_bg_color = (25, 120, 25)      # Deep green
            border_color = (50, 255, 100)       # Neon green
        else:
            status_text = f"{pose.direction.upper()}"
            badge_bg_color = (20, 20, 130)      # Deep red/amber
            border_color = (50, 70, 255)        # Alert red

        angle_text = f"Y:{pose.yaw:+.0f}\u00b0 P:{pose.pitch:+.0f}\u00b0 R:{pose.roll:+.0f}\u00b0"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        (tw1, th1), _ = cv2.getTextSize(status_text, font, font_scale, thickness)
        (tw2, th2), _ = cv2.getTextSize(angle_text, font, font_scale, thickness)
        box_w = max(tw1, tw2) + 16
        box_h = th1 + th2 + 16

        x1 = badge_x
        y1 = max(10, badge_y - box_h)
        x2 = min(w - 10, x1 + box_w)
        y2 = y1 + box_h

        # Semi-transparent background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), badge_bg_color, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), border_color, 1)
        cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)

        # Draw text labels
        cv2.putText(
            canvas,
            status_text,
            (x1 + 8, y1 + th1 + 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            angle_text,
            (x1 + 8, y1 + th1 + th2 + 10),
            font,
            font_scale,
            (200, 220, 255),
            thickness,
            cv2.LINE_AA,
        )

    return canvas
