"""Unit tests for 3D Head Pose and Attention Estimation."""

from pathlib import Path
import cv2
import numpy as np
import pytest

import detekface
from detekface.pose import (
    DEFAULT_3D_MODEL_POINTS,
    HeadPose,
    HeadPoseEstimator,
    draw_head_pose,
    estimate_pose,
)
from detekface.types import BoundingBox, FaceDetection, FaceLandmarks, Point

ASSETS_DIR = Path(__file__).parent / "assets"
SAMPLE_FACE = ASSETS_DIR / "sample_face.jpg"


def test_estimator_initialization():
    estimator = HeadPoseEstimator()
    assert estimator.yaw_threshold == 15.0
    assert estimator.pitch_threshold == 15.0
    assert estimator.roll_threshold == 20.0
    assert estimator.model_points.shape == (5, 3)

    custom = HeadPoseEstimator(yaw_threshold=25.0, pitch_threshold=10.0, roll_threshold=30.0)
    assert custom.yaw_threshold == 25.0
    assert custom.pitch_threshold == 10.0
    assert custom.roll_threshold == 30.0


def test_pose_dataclass_methods():
    pose = HeadPose(
        pitch=5.2,
        yaw=-12.4,
        roll=1.1,
        direction="Looking Right",
        is_looking_at_camera=True,
        attention_state="Attentive",
    )
    d = pose.as_dict()
    assert d["pitch"] == 5.2
    assert d["yaw"] == -12.4
    assert d["direction"] == "Looking Right"
    assert d["is_looking_at_camera"] is True
    assert d["attention_state"] == "Attentive"


def test_estimate_pose_with_facelandmarks():
    landmarks = FaceLandmarks(
        right_eye=Point(272, 269),
        left_eye=Point(328, 276),
        nose_tip=Point(310, 313),
        right_mouth=Point(271, 342),
        left_mouth=Point(311, 348),
    )
    pose = estimate_pose(landmarks, image_shape=(480, 640))
    assert isinstance(pose, HeadPose)
    assert isinstance(pose.pitch, float)
    assert isinstance(pose.yaw, float)
    assert isinstance(pose.roll, float)
    assert isinstance(pose.is_looking_at_camera, bool)
    assert pose.direction in ["Center", "Looking Left", "Looking Right", "Looking Up", "Looking Down"] or "Looking" in pose.direction
    assert pose.attention_state in ["Attentive", "Looking Away"]
    assert pose.rvec is not None
    assert pose.tvec is not None


def test_estimate_pose_with_tuples_and_points():
    tuples = [(272, 269), (328, 276), (310, 313), (271, 342), (311, 348)]
    pose1 = estimate_pose(tuples, image_shape=(480, 640))
    assert isinstance(pose1, HeadPose)

    points = [Point(x, y) for x, y in tuples]
    pose2 = estimate_pose(points, image_shape=(480, 640))
    assert isinstance(pose2, HeadPose)
    assert round(pose1.yaw, 2) == round(pose2.yaw, 2)


def test_estimate_pose_invalid_landmarks():
    estimator = HeadPoseEstimator()
    with pytest.raises(ValueError, match="Expected exactly 5 landmarks"):
        estimator.estimate([(10, 10), (20, 20)], image_shape=(480, 640))

    with pytest.raises(ValueError, match="Unsupported landmark point format"):
        estimator.estimate(["invalid"] * 5, image_shape=(480, 640))


def test_project_axes():
    estimator = HeadPoseEstimator()
    landmarks = FaceLandmarks(
        right_eye=Point(272, 269),
        left_eye=Point(328, 276),
        nose_tip=Point(310, 313),
        right_mouth=Point(271, 342),
        left_mouth=Point(311, 348),
    )
    pose = estimator.estimate(landmarks, image_shape=(480, 640))
    axes = estimator.project_axes(pose, image_shape=(480, 640), axis_length=50.0)

    assert axes is not None
    assert "origin" in axes
    assert "x_axis" in axes
    assert "y_axis" in axes
    assert "z_axis" in axes

    for key in ["origin", "x_axis", "y_axis", "z_axis"]:
        pt = axes[key]
        assert isinstance(pt, tuple)
        assert len(pt) == 2
        assert isinstance(pt[0], int) and isinstance(pt[1], int)


def test_draw_head_pose_rendering():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks = FaceLandmarks(
        right_eye=Point(272, 269),
        left_eye=Point(328, 276),
        nose_tip=Point(310, 313),
        right_mouth=Point(271, 342),
        left_mouth=Point(311, 348),
    )
    pose = estimate_pose(landmarks, (480, 640))
    bbox = BoundingBox(x=250, y=240, width=100, height=120)

    annotated = draw_head_pose(img, pose, bbox=bbox, show_axes=True, show_badge=True)
    assert annotated.shape == img.shape
    assert annotated.dtype == img.dtype
    # Ensure drawing occurred
    assert not np.array_equal(annotated, img)


def test_face_detection_and_result_integration():
    if not SAMPLE_FACE.exists():
        pytest.skip(f"Test sample asset {SAMPLE_FACE} missing.")

    result = detekface.detect(SAMPLE_FACE)
    assert result.has_faces

    face = result.faces[0]
    assert face.landmarks is not None

    # Test FaceDetection.estimate_pose and .pose
    pose = face.estimate_pose(image_shape=result.image.shape[:2])
    assert isinstance(pose, HeadPose)
    assert face.pose is not None

    # Test DetectionResult.estimate_poses
    poses = result.estimate_poses()
    assert len(poses) == result.count
    assert all(isinstance(p, HeadPose) for p in poses)

    # Test DetectionResult.annotate with show_pose=True
    annotated = result.annotate(shape="hud", show_pose=True)
    assert annotated.shape == result.image.shape
    assert not np.array_equal(annotated, result.image)
