# DetekFace Examples

This directory contains complete, runnable example scripts demonstrating key capabilities of `detekface`.

---

## Example Scripts Overview

| Script | Command | Description |
| :--- | :--- | :--- |
| **`webcam.py`** | `python examples/webcam.py` | Opens a live interactive webcam preview with real-time HUD overlays, smoothed FPS counter, face count badge, and keyboard face capture (`C`). |
| **`detect_image.py`** | `python examples/detect_image.py` | Detects faces in an image file, prints coordinates and 5 facial landmarks, and saves an annotated output image. |
| **`crop_face.py`** | `python examples/crop_face.py` | Programmatically crops the most prominent face from a photo and saves the unannotated crop to disk. |
| **`video_detection.py`** | `python examples/video_detection.py` | Processes a prerecorded video file frame-by-frame with customizable overlay styles and exports an annotated video. |
| **`benchmark.py`** | `python examples/benchmark.py` | Benchmarks inference latency (Mean, Median, P95) and FPS throughput across multiple resolutions (320x240, 640x480, 1280x720) on your local hardware. |
| **`head_pose.py`** | `python examples/head_pose.py` | Real-time 3D head pose orientation (Yaw, Pitch, Roll), gaze direction, and attention monitoring with 3D projected axes. |
| **`privacy_blur.py`** | `python examples/privacy_blur.py` | Live face privacy protection supporting 8-bit mosaic pixelation, frosted Gaussian blur, and eye redaction bars. |
| **`center_stage.py`** | `python examples/center_stage.py` | Virtual AI cameraman that smoothly pans and zooms to keep you cinematically centered and framed in real time. |
| **`terminal_face.py`** | `python examples/terminal_face.py` | Live streaming face detector rendered entirely inside your console using 24-bit TrueColor or Matrix Phosphor Green ASCII. |

---

## Running the Examples

Ensure your virtual environment is active before running any example:

```powershell
# Activate virtual environment on Windows
.\.venv\Scripts\Activate.ps1
```

### 1. Live Webcam Feed
```powershell
python examples/webcam.py
```
*Keyboard controls:* Press **`C`** to capture a face, or **`Q`** / **`Esc`** to quit.

### 2. Still Image Detection with Custom Style
```powershell
python examples/detect_image.py tests/assets/sample_face.jpg --save output.jpg --shape hud --color "#00FF88" --show-confidence
```

### 3. Programmatic Face Cropping
```powershell
python examples/crop_face.py tests/assets/sample_face.jpg
```

### 4. Video File Processing
```powershell
python examples/video_detection.py path/to/clip.mp4 --output annotated_clip.mp4 --shape rounded --color cyan
```

### 5. Performance Benchmark
```powershell
python examples/benchmark.py
```

### 6. Live 3D Head Pose & Attention Tracking
```powershell
python examples/head_pose.py
```
*Keyboard controls:* Press **`A`** to toggle 3D coordinate axes, **`H`** to toggle HUD attention badge, or **`Q`** / **`Esc`** to quit.

### 7. Live Privacy & Face Anonymization
```powershell
python examples/privacy_blur.py
```
*Keyboard controls:* Press **`1`** for Pixelate/Mosaic, **`2`** for Gaussian Blur, **`3`** for Eye Censor Bar, **`4`** to turn off, or **`Q`** / **`Esc`** to quit.

### 8. Live "Center Stage" AI Cameraman
```powershell
python examples/center_stage.py
```
*Keyboard controls:* Press **`P`** to toggle Radar thumbnail, **`R`** to reset zoom, **`+`** / **`-`** to adjust tracking speed, or **`Q`** / **`Esc`** to quit.

### 9. Terminal ASCII / Matrix Face Streamer
```powershell
python examples/terminal_face.py
```
*Keyboard controls:* Press **`M`** for Matrix Green, **`C`** for 24-bit TrueColor, **`G`** for Grayscale, or **`Q`** to quit.


