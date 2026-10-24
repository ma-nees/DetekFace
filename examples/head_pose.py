"""Live webcam head pose estimation & attention tracking example using detekface.

Demonstrates real-time 3D orientation (Yaw, Pitch, Roll), gaze direction,
and attention state (Attentive vs Looking Away) with 3D projected coordinate axes.
"""

import argparse
import sys
import time
import cv2
import detekface
from detekface.pose import HeadPoseEstimator, draw_head_pose


def main() -> None:
    parser = argparse.ArgumentParser(description="Live 3D Head Pose & Attention Tracking with detekface.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirroring of the webcam stream.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.65,
        help="Detection confidence threshold between 0.0 and 1.0 (default: 0.65).",
    )
    parser.add_argument(
        "--yaw-threshold",
        type=float,
        default=15.0,
        help="Yaw threshold in degrees for looking away (default: 15.0).",
    )
    parser.add_argument(
        "--pitch-threshold",
        type=float,
        default=15.0,
        help="Pitch threshold in degrees for looking away (default: 15.0).",
    )
    parser.add_argument(
        "--roll-threshold",
        type=float,
        default=20.0,
        help="Roll threshold in degrees for head tilt (default: 20.0).",
    )

    args = parser.parse_args()

    estimator = HeadPoseEstimator(
        yaw_threshold=args.yaw_threshold,
        pitch_threshold=args.pitch_threshold,
        roll_threshold=args.roll_threshold,
    )

    print("==========================================================")
    print(" DetekFace: 3D Head Pose & Attention Tracking Demo")
    print("==========================================================")
    print(f"Camera index     : {args.camera_index}")
    print(f"Yaw threshold    : \u00b1{args.yaw_threshold}\u00b0")
    print(f"Pitch threshold  : \u00b1{args.pitch_threshold}\u00b0")
    print("Controls:")
    print("  [A] Toggle 3D orientation axes")
    print("  [H] Toggle HUD attention badge")
    print("  [Q] or [Escape] Exit preview")
    print("==========================================================\n")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {args.camera_index}.")
        sys.exit(1)

    show_axes = True
    show_badge = True
    fps_counter = detekface.FPSCounter(window_size=30)
    window_name = "DetekFace - 3D Head Pose & Attention Tracker"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Failed to grab frame from camera.")
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            fps = fps_counter.tick()

            # Detect faces with landmarks
            result = detekface.detect(frame, confidence_threshold=args.confidence)

            # Draw standard subtle bounding box
            annotated = result.annotate(shape="corners", color="#00FF88", thickness=2)

            # Draw 3D pose, axes, and attention badge for each face
            for face in result.faces:
                if face.landmarks is not None:
                    pose = estimator.estimate(face.landmarks, (h, w))
                    annotated = draw_head_pose(
                        annotated,
                        pose,
                        bbox=face.bbox,
                        show_axes=show_axes,
                        show_badge=show_badge,
                        estimator=estimator,
                    )

            # Draw top FPS and face count metric badge
            detekface.draw_metrics_overlay(annotated, fps=fps, face_count=result.count)

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' or Esc
                break
            elif key in (ord("a"), ord("A")):
                show_axes = not show_axes
            elif key in (ord("h"), ord("H")):
                show_badge = not show_badge

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nHead pose tracking preview closed.")


if __name__ == "__main__":
    main()
