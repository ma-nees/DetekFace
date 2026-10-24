"""Unified DetekFace CLI and interactive terminal dashboard."""

import argparse
import os
import subprocess
import sys
import time
from typing import Optional
import cv2
import detekface
from detekface.autocam import CenterStageTracker
from detekface.pose import HeadPoseEstimator, draw_head_pose

BANNER = r"""
  ____       _       _     _____             
 |  _ \  ___| |_ ___| | __|  ___|_ _  ___ ___ 
 | | | |/ _ \ __/ _ \ |/ /| |_ / _` |/ __/ _ \
 | |_| |  __/ ||  __/   < |  _| (_| | (_|  __/
 |____/ \___|\__\___|_|\_\|_|  \__,_|\___\___|
        Face Detection, 3D Pose, Privacy & AI Cameraman Hub
"""

MENU_TEXT = """
\033[1;36m========================================================================\033[0m
\033[1;32m                  DETEKFACE ALL-IN-ONE COMMAND HUB                      \033[0m
\033[1;36m========================================================================\033[0m

\033[1mSelect an experience to launch:\033[0m

  \033[1;33m[0]\033[0m  \033[1mAll-In-One Studio\033[0m       (Pose + Privacy + Center Stage + HUD in ONE live feed)
  \033[1;33m[1]\033[0m  \033[1mClassic Live Webcam\033[0m     (Interactive preview, HUD badges, capture 'C')
  \033[1;33m[2]\033[0m  \033[1m3D Head Pose Tracker\033[0m    (Yaw/Pitch/Roll, 3D coordinate axes, attention)
  \033[1;33m[3]\033[0m  \033[1mPrivacy & Anonymizer\033[0m    (Pixelate, frosted Gaussian blur, eye censor bar)
  \033[1;33m[4]\033[0m  \033[1mCenter Stage AI Cam\033[0m     (Cinematic auto-zoom and face tracking cameraman)
  \033[1;33m[5]\033[0m  \033[1mTerminal ASCII Stream\033[0m   (Live 24-bit TrueColor / Matrix ASCII video)
  \033[1;33m[6]\033[0m  \033[1mHardware Benchmark\033[0m      (Inference throughput & FPS across resolutions)
  \033[1;33m[7]\033[0m  \033[1mSingle Image Detect\033[0m     (Annotate sample photo with custom style)
  \033[1;33m[8]\033[0m  \033[1mAuto Face Crop\033[0m          (Extract highest-confidence face to file)

  \033[1;31m[Q]\033[0m  \033[1mQuit\033[0m

\033[1;36m========================================================================\033[0m
"""


def run_all_in_one_studio(camera_index: int = 0, mirror: bool = True) -> None:
    """Run the Ultimate Live Studio with all features toggleable in a single webcam feed."""
    print("\n" + "=" * 65)
    print(" DETEKFACE ALL-IN-ONE LIVE STUDIO")
    print("=" * 65)
    print("Active Keyboard Hotkeys (press inside the preview window):")
    print("  [P]  Toggle 3D Head Pose & 3D Axes")
    print("  [A]  Cycle Privacy: OFF -> Pixelate -> Blur -> Censor Bar")
    print("  [C]  Toggle Center Stage Auto-Zoom Cameraman")
    print("  [S]  Cycle Overlay Shapes: HUD -> Corners -> Rounded -> Rectangle")
    print("  [K]  Capture Face Crop to disk")
    print("  [H]  Toggle Top Metrics HUD (FPS / Face Count)")
    print("  [Q] or [Esc] Exit Studio")
    print("=" * 65 + "\n")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return

    tracker = CenterStageTracker(smoothing=0.09, max_zoom=2.6, show_pip=True)
    pose_estimator = HeadPoseEstimator()
    fps_counter = detekface.FPSCounter(window_size=30)

    # Feature state flags
    show_pose = True
    privacy_mode = "off"  # 'off', 'pixelate', 'blur', 'censor_eyes'
    enable_center_stage = True
    current_shape = "hud"
    show_metrics = True
    capture_count = 0

    shape_cycle = ["hud", "corners", "rounded", "rectangle", "circle"]
    privacy_cycle = ["off", "pixelate", "blur", "censor_eyes"]

    window_name = "DetekFace All-In-One Studio"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            fps = fps_counter.tick()

            # Detect faces
            result = detekface.detect(frame, confidence_threshold=0.65)

            # 1. Center Stage framing
            if enable_center_stage:
                display_frame, zoom = tracker.process_frame(frame, result.faces)
            else:
                display_frame = frame.copy()
                zoom = 1.0

            # 2. Privacy Anonymization
            if privacy_mode != "off" and result.has_faces:
                display_frame = detekface.anonymize(display_frame, result.faces, mode=privacy_mode)

            # 3. Face Bounding Box & HUD styling (if not full blur)
            if privacy_mode in ("off", "censor_eyes"):
                display_frame = detekface.draw_faces(
                    display_frame,
                    result.faces,
                    shape=current_shape,
                    color="#00FF88",
                    thickness=2,
                    show_confidence=True,
                    copy=False,
                )

            # 4. 3D Head Pose & Axes
            if show_pose and privacy_mode != "blur":
                h, w = display_frame.shape[:2]
                for face in result.faces:
                    if face.landmarks is not None:
                        pose = pose_estimator.estimate(face.landmarks, (h, w))
                        display_frame = draw_head_pose(
                            display_frame,
                            pose,
                            bbox=face.bbox,
                            show_axes=True,
                            show_badge=True,
                            estimator=pose_estimator,
                        )

            # 5. Top Status Metrics HUD
            if show_metrics:
                detekface.draw_metrics_overlay(display_frame, fps=fps, face_count=result.count)

            # 6. Bottom Feature State HUD Bar
            h_disp, w_disp = display_frame.shape[:2]
            status_items = [
                f"CAM: {'ON' if enable_center_stage else 'OFF'} ({zoom:.1f}x)",
                f"POSE: {'ON' if show_pose else 'OFF'}",
                f"PRIVACY: {privacy_mode.upper()}",
                f"SHAPE: {current_shape.upper()}",
            ]
            hud_line = " | ".join(status_items)
            cv2.rectangle(display_frame, (0, h_disp - 28), (w_disp, h_disp), (15, 15, 15), -1)
            cv2.putText(
                display_frame,
                hud_line,
                (12, h_disp - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 136),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("p"), ord("P")):
                show_pose = not show_pose
            elif key in (ord("a"), ord("A")):
                idx = (privacy_cycle.index(privacy_mode) + 1) % len(privacy_cycle)
                privacy_mode = privacy_cycle[idx]
            elif key in (ord("c"), ord("C")):
                enable_center_stage = not enable_center_stage
            elif key in (ord("s"), ord("S")):
                s_idx = (shape_cycle.index(current_shape) + 1) % len(shape_cycle)
                current_shape = shape_cycle[s_idx]
            elif key in (ord("h"), ord("H")):
                show_metrics = not show_metrics
            elif key in (ord("k"), ord("K")):
                if result.has_faces:
                    os.makedirs("captures", exist_ok=True)
                    crop_path = f"captures/studio_face_{int(time.time())}.jpg"
                    cropped = result.faces[0].bbox.crop(frame) if hasattr(result.faces[0].bbox, "crop") else frame
                    cv2.imwrite(crop_path, cropped)
                    capture_count += 1
                    print(f"[STUDIO] Face captured to: {crop_path}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nAll-In-One Studio closed.")


def main() -> None:
    """CLI Entry point for detekface command."""
    parser = argparse.ArgumentParser(
        prog="detekface",
        description="DetekFace Command Hub: Face Detection, 3D Pose, Privacy & Center Stage.",
    )
    parser.add_argument(
        "feature",
        nargs="?",
        default=None,
        choices=[
            "studio",
            "webcam",
            "pose",
            "privacy",
            "center-stage",
            "ascii",
            "benchmark",
            "image",
            "crop",
        ],
        help="Feature to launch directly (or omit to open interactive menu).",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--no-mirror", action="store_true", help="Disable horizontal mirroring.")

    args = parser.parse_args()

    # Direct launch by argument
    if args.feature == "studio":
        run_all_in_one_studio(camera_index=args.camera_index, mirror=not args.no_mirror)
        return
    elif args.feature == "webcam":
        subprocess.run([sys.executable, "examples/webcam.py"])
        return
    elif args.feature == "pose":
        subprocess.run([sys.executable, "examples/head_pose.py"])
        return
    elif args.feature == "privacy":
        subprocess.run([sys.executable, "examples/privacy_blur.py"])
        return
    elif args.feature == "center-stage":
        subprocess.run([sys.executable, "examples/center_stage.py"])
        return
    elif args.feature == "ascii":
        subprocess.run([sys.executable, "examples/terminal_face.py"])
        return
    elif args.feature == "benchmark":
        subprocess.run([sys.executable, "examples/benchmark.py"])
        return
    elif args.feature == "image":
        subprocess.run([sys.executable, "examples/detect_image.py", "tests/assets/sample_face.jpg", "--save", "output.jpg", "--shape", "hud"])
        return
    elif args.feature == "crop":
        subprocess.run([sys.executable, "examples/crop_face.py", "tests/assets/sample_face.jpg"])
        return

    # Interactive Terminal Dashboard
    if os.name == "nt":
        os.system("")

    while True:
        print(BANNER)
        print(MENU_TEXT)
        choice = input("\033[1;32mEnter your choice [0-8, Q]: \033[0m").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("\nGoodbye!\n")
            break
        elif choice == "0":
            run_all_in_one_studio(camera_index=args.camera_index, mirror=not args.no_mirror)
        elif choice == "1":
            subprocess.run([sys.executable, "examples/webcam.py"])
        elif choice == "2":
            subprocess.run([sys.executable, "examples/head_pose.py"])
        elif choice == "3":
            subprocess.run([sys.executable, "examples/privacy_blur.py"])
        elif choice == "4":
            subprocess.run([sys.executable, "examples/center_stage.py"])
        elif choice == "5":
            subprocess.run([sys.executable, "examples/terminal_face.py"])
        elif choice == "6":
            subprocess.run([sys.executable, "examples/benchmark.py"])
        elif choice == "7":
            subprocess.run([sys.executable, "examples/detect_image.py", "tests/assets/sample_face.jpg", "--save", "output.jpg", "--shape", "hud"])
        elif choice == "8":
            subprocess.run([sys.executable, "examples/crop_face.py", "tests/assets/sample_face.jpg"])
        else:
            print("\033[1;31mInvalid selection, please try again.\033[0m")
            time.sleep(1)


if __name__ == "__main__":
    main()
