"""Live webcam Center Stage auto-framing and zoom tracking example using detekface.

Keeps you cinematically centered and properly framed in real time as you move around.
"""

import argparse
import sys
import cv2
import detekface
from detekface.autocam import CenterStageTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Center Stage auto-framing camera with detekface.")
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
        "--smoothing",
        type=float,
        default=0.08,
        help="Motion smoothing factor (0.02 = very slow, 0.2 = snappy). Default: 0.08.",
    )
    parser.add_argument(
        "--max-zoom",
        type=float,
        default=2.5,
        help="Maximum auto-zoom factor (default: 2.5x).",
    )
    parser.add_argument(
        "--no-pip",
        action="store_true",
        help="Disable the miniature Picture-in-Picture wide radar thumbnail.",
    )

    args = parser.parse_args()

    tracker = CenterStageTracker(
        smoothing=args.smoothing,
        max_zoom=args.max_zoom,
        show_pip=not args.no_pip,
    )

    print("==========================================================")
    print(" DetekFace: 'Center Stage' Virtual Cameraman Demo")
    print("==========================================================")
    print(f"Camera index : {args.camera_index}")
    print(f"Max zoom     : {args.max_zoom:.1f}x")
    print(f"Smoothing    : {args.smoothing}")
    print("Controls:")
    print("  [P] Toggle Picture-in-Picture Radar thumbnail")
    print("  [R] Reset camera framing to wide angle")
    print("  [+] Speed up camera tracking (snappier)")
    print("  [-] Slow down camera tracking (more cinematic)")
    print("  [Q] or [Escape] Exit")
    print("==========================================================\n")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_index}.")
        sys.exit(1)

    fps_counter = detekface.FPSCounter(window_size=30)
    window_name = "DetekFace - Center Stage AI Cameraman"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Failed to grab frame from camera.")
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            fps = fps_counter.tick()

            # Detect faces in full resolution
            result = detekface.detect(frame, confidence_threshold=0.6)

            # Apply Center Stage auto-framing
            framed, zoom = tracker.process_frame(frame, result.faces)

            # Draw HUD metrics: FPS & face count
            detekface.draw_metrics_overlay(framed, fps=fps, face_count=result.count)

            # Draw bottom zoom HUD status
            zoom_text = f"CENTER STAGE | {zoom:.1f}x ZOOM"
            cv2.putText(
                framed,
                zoom_text,
                (15, framed.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 136),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, framed)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("p"), ord("P")):
                tracker.show_pip = not tracker.show_pip
            elif key in (ord("r"), ord("R")):
                tracker.reset()
            elif key in (ord("+"), ord("=")):
                tracker.smoothing = min(0.35, tracker.smoothing + 0.02)
                print(f"[CAMERAMAN] Tracking smoothing adjusted to: {tracker.smoothing:.2f}")
            elif key in (ord("-"), ord("_")):
                tracker.smoothing = max(0.02, tracker.smoothing - 0.02)
                print(f"[CAMERAMAN] Tracking smoothing adjusted to: {tracker.smoothing:.2f}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nCenter Stage camera terminated.")


if __name__ == "__main__":
    main()
