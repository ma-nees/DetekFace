"""Head pose estimation module using 5 facial landmarks and Perspective-n-Point (solvePnP).

Computes 3D head orientation (Pitch, Yaw, Roll) and determines attention state / gaze direction.
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from detekface.types import FaceLandmarks, Point


# Standard 3D anthropometric face model coordinates (in millimeters, centered at nose tip)
# Points order: [Right Eye, Left Eye, Nose Tip, Right Mouth Corner, Left Mouth Corner]
# Coordinate system:
# X: points right (subject's left / viewer's right)
# Y: points down
# Z: points away from camera
DEFAULT_3D_MODEL_POINTS = np.array([
    [-30.0, -20.0, -10.0],  # Right Eye (viewer's left)
    [ 30.0, -20.0, -10.0],  # Left Eye (viewer's right)
    [  0.0,   0.0,   0.0],  # Nose Tip
    [-20.0,  30.0, -10.0],  # Right Mouth Corner
    [ 20.0,  30.0, -10.0],  # Left Mouth Corner
], dtype=np.float64)


@dataclass(frozen=True)
class HeadPose:
    """Estimated 3D orientation and attention status of a detected face.

    Attributes:
        pitch: Nodding angle in degrees (negative = looking down, positive = looking up).
        yaw: Turning angle in degrees (negative = looking right, positive = looking left).
        roll: In-plane tilting angle in degrees (ear-to-shoulder tilt).
        direction: Human-readable gaze direction (e.g. 'Center', 'Looking Left', 'Looking Right').
        is_looking_at_camera: True if the subject's face is oriented towards the camera.
        attention_state: 'Attentive' if looking at camera, otherwise 'Looking Away'.
        rvec: 3D rotation vector from solvePnP.
        tvec: 3D translation vector from solvePnP.
    """

    pitch: float
    yaw: float
    roll: float
    direction: str
    is_looking_at_camera: bool
    attention_state: str
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None

    def as_dict(self) -> dict:
        """Convert pose attributes to a dictionary representation."""
        return {
            "pitch": round(self.pitch, 2),
            "yaw": round(self.yaw, 2),
            "roll": round(self.roll, 2),
            "direction": self.direction,
            "is_looking_at_camera": self.is_looking_at_camera,
            "attention_state": self.attention_state,
        }


class HeadPoseEstimator:
    """Estimates head pose (pitch, yaw, roll) from 5 facial landmarks using cv2.solvePnP."""

    def __init__(
        self,
        yaw_threshold: float = 15.0,
        pitch_threshold: float = 15.0,
        roll_threshold: float = 20.0,
        model_points: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the HeadPoseEstimator.

        Args:
            yaw_threshold: Angular threshold in degrees beyond which the face is turned away horizontally.
            pitch_threshold: Angular threshold in degrees beyond which the face is turned away vertically.
            roll_threshold: Angular threshold in degrees beyond which the face is considered tilted.
            model_points: Optional custom 5x3 numpy array of canonical 3D face model coordinates.
        """
        self.yaw_threshold = float(yaw_threshold)
        self.pitch_threshold = float(pitch_threshold)
        self.roll_threshold = float(roll_threshold)
        self.model_points = (
            np.array(model_points, dtype=np.float64)
            if model_points is not None
            else DEFAULT_3D_MODEL_POINTS.copy()
        )

    def estimate(
        self,
        landmarks: Union[FaceLandmarks, Sequence[Point], Sequence[Tuple[int, int]]],
        image_shape: Tuple[int, int],
    ) -> HeadPose:
        """Estimate 3D head pose from 5 facial landmarks and image dimensions.

        Args:
            landmarks: FaceLandmarks instance, list of Points, or list of (x, y) tuples.
            image_shape: Tuple of (height, width) of the image frame.

        Returns:
            HeadPose containing pitch, yaw, roll, gaze direction, and attention status.
        """
        height, width = image_shape[:2]

        if isinstance(landmarks, FaceLandmarks):
            points_2d = np.array([
                [landmarks.right_eye.x, landmarks.right_eye.y],
                [landmarks.left_eye.x, landmarks.left_eye.y],
                [landmarks.nose_tip.x, landmarks.nose_tip.y],
                [landmarks.right_mouth.x, landmarks.right_mouth.y],
                [landmarks.left_mouth.x, landmarks.left_mouth.y],
            ], dtype=np.float64)
        else:
            extracted = []
            for p in landmarks:
                if isinstance(p, Point):
                    extracted.append([p.x, p.y])
                elif isinstance(p, (tuple, list)) and len(p) >= 2:
                    extracted.append([p[0], p[1]])
                else:
                    raise ValueError(f"Unsupported landmark point format: {p}")
            if len(extracted) != 5:
                raise ValueError(f"Expected exactly 5 landmarks, got {len(extracted)}")
            points_2d = np.array(extracted, dtype=np.float64)

        # Approximate camera intrinsic matrix
        focal_length = float(width)
        center = (float(width) / 2.0, float(height) / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve Perspective-n-Point
        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not success:
            return HeadPose(
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
                direction="Unknown",
                is_looking_at_camera=False,
                attention_state="Unknown",
                rvec=None,
                tvec=None,
            )

        # Convert rotation vector to rotation matrix and extract Euler angles
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Direction categorization
        horiz = ""
        vert = ""

        if yaw > self.yaw_threshold:
            horiz = "Left"
        elif yaw < -self.yaw_threshold:
            horiz = "Right"

        if pitch > self.pitch_threshold:
            vert = "Up"
        elif pitch < -self.pitch_threshold:
            vert = "Down"

        if not horiz and not vert:
            direction = "Center"
        elif horiz and vert:
            direction = f"Looking {vert}-{horiz}"
        else:
            direction = f"Looking {horiz or vert}"

        is_looking = (
            abs(yaw) <= self.yaw_threshold
            and abs(pitch) <= self.pitch_threshold
            and abs(roll) <= self.roll_threshold
        )
        attention_state = "Attentive" if is_looking else "Looking Away"

        return HeadPose(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            direction=direction,
            is_looking_at_camera=is_looking,
            attention_state=attention_state,
            rvec=rvec,
            tvec=tvec,
        )

    def project_axes(
        self,
        pose: HeadPose,
        image_shape: Tuple[int, int],
        axis_length: float = 50.0,
    ) -> Optional[dict]:
        """Project 3D axes (X, Y, Z) onto the image plane for visualization.

        Args:
            pose: Computed HeadPose containing rvec and tvec.
            image_shape: Tuple of (height, width).
            axis_length: Length of 3D axis arrows in millimeters.

        Returns:
            Dict containing 2D points for nose origin, x_axis, y_axis, and z_axis.
        """
        if pose.rvec is None or pose.tvec is None:
            return None

        height, width = image_shape[:2]
        focal_length = float(width)
        center = (float(width) / 2.0, float(height) / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 3D points originating at nose (0, 0, 0)
        axes_3d = np.array([
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],   # X-axis (Right / Red)
            [0.0, axis_length, 0.0],   # Y-axis (Down / Green)
            [0.0, 0.0, axis_length],   # Z-axis (Forward Gaze / Blue)
        ], dtype=np.float64)

        projected, _ = cv2.projectPoints(axes_3d, pose.rvec, pose.tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)

        return {
            "origin": (int(round(projected[0][0])), int(round(projected[0][1]))),
            "x_axis": (int(round(projected[1][0])), int(round(projected[1][1]))),
            "y_axis": (int(round(projected[2][0])), int(round(projected[2][1]))),
            "z_axis": (int(round(projected[3][0])), int(round(projected[3][1]))),
        }


# Module-level default estimator instance
_default_estimator = HeadPoseEstimator()


def estimate_pose(
    landmarks: Union[FaceLandmarks, Sequence[Point], Sequence[Tuple[int, int]]],
    image_shape: Tuple[int, int],
    yaw_threshold: float = 15.0,
    pitch_threshold: float = 15.0,
    roll_threshold: float = 20.0,
) -> HeadPose:
    """Convenience function to estimate 3D head pose from facial landmarks.

    Args:
        landmarks: Facial landmarks detected by detekface.
        image_shape: (height, width) of the target frame.
        yaw_threshold: Horizontal turn threshold in degrees.
        pitch_threshold: Vertical nod threshold in degrees.
        roll_threshold: Tilt threshold in degrees.

    Returns:
        HeadPose instance with pitch, yaw, roll, direction, and attention status.
    """
    if yaw_threshold != 15.0 or pitch_threshold != 15.0 or roll_threshold != 20.0:
        estimator = HeadPoseEstimator(
            yaw_threshold=yaw_threshold,
            pitch_threshold=pitch_threshold,
            roll_threshold=roll_threshold,
        )
        return estimator.estimate(landmarks, image_shape)
    return _default_estimator.estimate(landmarks, image_shape)
