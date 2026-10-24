"""Live terminal ASCII face streamer using detekface.

Streams live webcam video and face tracking directly inside the console in:
  [M] Matrix Phosphor Green
  [C] 24-bit TrueColor
  [G] Plain Grayscale
"""

import argparse
import os
import sys
import time
import cv2
import detekface
from detekface.ascii import frame_to_ascii

# Windows non-blocking keyboard input
try:
    import msvcrt
except ImportError:
    msvcrt = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Terminal ASCII Face Streamer with detekface.")
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
        default="matrix",
        choices=["matrix", "color", "gray"],
        help="Initial ASCII rendering mode: 'matrix', 'color', or 'gray' (default: 'matrix').",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Target column width (default: auto-fit terminal).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=25.0,
        help="Target rendering framerate (default: 25.0 FPS).",
    )

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_index}.")
        sys.exit(1)

    # Enable ANSI escape sequences on Windows console
    if os.name == "nt":
        os.system("")

    current_mode = args.mode
    fps_counter = detekface.FPSCounter(window_size=20)
    frame_interval = 1.0 / max(1.0, args.target_fps)

    # Clear screen and hide cursor
    sys.stdout.write("\033[2J\033[?25l")
    sys.stdout.flush()

    try:
        while True:
            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            fps = fps_counter.tick()

            # Detect faces
            result = detekface.detect(frame, confidence_threshold=0.65)

            # Generate ASCII representation
            ascii_frame = frame_to_ascii(
                frame,
                faces=result.faces,
                cols=args.cols,
                mode=current_mode,
            )

            # Build status banner
            header = (
                f"\033[1;37;44m DETEKFACE TERMINAL STREAM | {fps:.1f} FPS | "
                f"FACES: {result.count} | MODE: {current_mode.upper()} \033[0m\n"
                f"Keys: [M] Matrix | [C] Color | [G] Gray | [Q] Quit\n"
            )

            # Write frame using cursor home to prevent screen flicker
            sys.stdout.write("\033[H" + header + ascii_frame + "\n")
            sys.stdout.flush()

            # Handle non-blocking keypresses on Windows
            if msvcrt and msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b"q", b"Q", b"\x1b", b"\x03"):  # 'q', Esc, Ctrl+C
                    break
                elif key in (b"m", b"M"):
                    current_mode = "matrix"
                elif key in (b"c", b"C"):
                    current_mode = "color"
                elif key in (b"g", b"G"):
                    current_mode = "gray"

            elapsed = time.perf_counter() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        # Restore cursor and clear screen
        sys.stdout.write("\033[?25h\033[2J\033[H")
        sys.stdout.flush()
        print("Terminal ASCII stream ended.")


if __name__ == "__main__":
    main()
