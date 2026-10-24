"""Live webcam face privacy & anonymization example using detekface.

Demonstrates real-time identity protection using:
  [1] 8-bit retro mosaic pixelation
  [2] Gaussian frosted blur
  [3] Dynamic eye redaction bar
  [4] Anonymization off (raw feed)
"""

import argparse
import sys
import cv2
import detekface


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam face anonymization with detekface.")
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
        "--mode",
        type=str,
        default="pixelate",
        choices=["pixelate", "blur", "censor_eyes"],
        help="Default anonymization mode (default: 'pixelate').",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block size in pixels for pixelate mode (default: 16).",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=51,
        help="Kernel size for blur mode (default: 51).",
    )

    args = parser.parse_args()

    print("==========================================================")
    print(" DetekFace: Real-Time Face Privacy & Anonymizer Demo")
    print("==========================================================")
    print(f"Camera index : {args.camera_index}")
    print(f"Active mode  : {args.mode}")
    print("Interactive Controls:")
    print("  [1] Pixelate / Mosaic Mode")
    print("  [2] Gaussian Blur Mode")
    print("  [3] Eye Redaction Censor Bar Mode")
    print("  [4] Anonymization OFF")
    print("  [Q] or [Escape] Exit")
    print("==========================================================\n")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_index}.")
        sys.exit(1)

    current_mode = args.mode
    enabled = True
    fps_counter = detekface.FPSCounter(window_size=30)
    window_name = "DetekFace - Face Privacy Anonymizer"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Failed to grab frame from camera.")
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            fps = fps_counter.tick()

            # Detect faces
            result = detekface.detect(frame, confidence_threshold=0.65)

            if enabled and result.has_faces:
                output = result.anonymize(
                    mode=current_mode,
                    block_size=args.block_size,
                    kernel_size=args.kernel_size,
                )
            else:
                output = frame.copy()

            # Draw status badge in the top-left corner
            mode_display = f"PRIVACY: {current_mode.upper()}" if enabled else "PRIVACY: OFF"
            detekface.draw_metrics_overlay(output, fps=fps, face_count=result.count)

            # Draw active mode HUD tag
            cv2.putText(
                output,
                mode_display,
                (15, output.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 136) if enabled else (150, 150, 150),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, output)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == ord("1"):
                current_mode = "pixelate"
                enabled = True
            elif key == ord("2"):
                current_mode = "blur"
                enabled = True
            elif key == ord("3"):
                current_mode = "censor_eyes"
                enabled = True
            elif key == ord("4"):
                enabled = False

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nPrivacy preview closed.")


if __name__ == "__main__":
    main()
