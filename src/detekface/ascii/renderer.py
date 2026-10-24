"""Terminal ASCII face renderer for streaming camera frames and face detections directly in the console."""

import os
import shutil
from typing import Optional, Sequence, Tuple
import cv2
import numpy as np

from detekface.types import FaceDetection

ASCII_RAMP_SHORT = " .:-=+*#%@"
ASCII_RAMP_BLOCKS = " ░▒▓█"
ASCII_RAMP_DETAILED = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"


def frame_to_ascii(
    frame: np.ndarray,
    faces: Optional[Sequence[FaceDetection]] = None,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
    mode: str = "color",
    ramp: str = ASCII_RAMP_SHORT,
) -> str:
    """Convert an image frame and face detections into an ANSI ASCII character grid.

    Args:
        frame: BGR image array.
        faces: Optional sequence of FaceDetection instances to highlight with ASCII borders.
        cols: Target character columns (defaults to console width or 80).
        rows: Target character rows (defaults to console height or 35).
        mode: 'color' (24-bit ANSI TrueColor), 'matrix' (phosphor green), or 'gray' (plain ASCII).
        ramp: String of ASCII characters ordered from darkest to brightest luminance.

    Returns:
        String ready to be printed to the terminal.
    """
    term_cols, term_rows = shutil.get_terminal_size(fallback=(80, 35))
    target_cols = cols or min(100, max(40, term_cols - 2))
    # Account for terminal character aspect ratio (characters are typically ~2x taller than wide)
    char_aspect = 0.5
    h, w = frame.shape[:2]
    img_aspect = h / w

    if rows is not None:
        target_rows = rows
    else:
        target_rows = max(15, min(term_rows - 4, int(target_cols * img_aspect * char_aspect)))

    # Resize image to target ASCII resolution
    small_bgr = cv2.resize(frame, (target_cols, target_rows), interpolation=cv2.INTER_AREA)
    small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

    # Calculate scale factor for mapping face bounding boxes to ASCII grid
    scale_x = target_cols / float(w)
    scale_y = target_rows / float(h)

    # Build face boundary grid set for fast lookup
    face_boxes = []
    if faces:
        for face in faces:
            bx1 = int(round(face.bbox.x1 * scale_x))
            by1 = int(round(face.bbox.y1 * scale_y))
            bx2 = int(round(face.bbox.x2 * scale_x))
            by2 = int(round(face.bbox.y2 * scale_y))
            face_boxes.append((bx1, by1, bx2, by2))

    num_chars = len(ramp)
    lines = []
    mode_clean = mode.lower().strip()

    for r in range(target_rows):
        line_chars = []
        for c in range(target_cols):
            # Check if this cell is part of a face bounding box border
            is_face_corner = False
            is_face_horiz = False
            is_face_vert = False

            for (bx1, by1, bx2, by2) in face_boxes:
                if (c in (bx1, bx2) and r in (by1, by2)):
                    is_face_corner = True
                    break
                elif (c in (bx1, bx2) and by1 <= r <= by2):
                    is_face_vert = True
                    break
                elif (r in (by1, by2) and bx1 <= c <= bx2):
                    is_face_horiz = True
                    break

            if is_face_corner:
                ch = "+"
            elif is_face_horiz:
                ch = "-"
            elif is_face_vert:
                ch = "|"
            else:
                gray_val = small_gray[r, c]
                char_idx = int((gray_val / 255.0) * (num_chars - 1))
                ch = ramp[char_idx]

            # Color formatting
            if is_face_corner or is_face_horiz or is_face_vert:
                # Face reticle glows vibrant cyan/green
                line_chars.append(f"\033[38;2;0;255;136m\033[1m{ch}\033[0m")
            elif mode_clean == "color":
                b, g, red = small_bgr[r, c]
                line_chars.append(f"\033[38;2;{red};{g};{b}m{ch}\033[0m")
            elif mode_clean == "matrix":
                gray_val = small_gray[r, c]
                # Matrix phosphor green tint based on luminance
                line_chars.append(f"\033[38;2;0;{max(40, gray_val)};{int(gray_val * 0.3)}m{ch}\033[0m")
            else:
                line_chars.append(ch)

        lines.append("".join(line_chars))

    return "\n".join(lines)
