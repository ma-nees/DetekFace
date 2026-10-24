# DetekFace Documentation Index

<p align="center">
  <img src="assets/detekface_main_logo.png" alt="DetekFace Logo" width="500" />
</p>

Welcome to the comprehensive documentation for **DetekFace**, a clean, beginner-friendly Python package for real-time and static face detection powered by YuNet and OpenCV.

---

## Documentation Navigation

| Document | Topic & Focus | Best For |
| :--- | :--- | :--- |
| **[Installation](installation.md)** | System requirements, virtual environment setup on Windows 11, and OpenCV GUI considerations. | Getting started, troubleshooting environment issues |
| **[Quick Start](quickstart.md)** | Core concepts (bounding boxes, confidence scores, landmarks, FPS) and minimal runnable examples. | Beginners learning computer vision |
| **[Command Hub & Studio](../run.py)** | Single-command launcher (`python run.py`) and All-In-One Studio combining all features into one webcam feed. | Quick experimentation, full-featured demo |
| **[Webcam Streaming](webcam.md)** | Real-time camera feed (`detekface.camera()`), keyboard shortcuts (`Q`, `Esc`, `C`), live overlays, and callbacks. | Interactive apps, kiosks, webcam tools |
| **[3D Head Pose & Gaze](../examples/head_pose.py)** | 3D Euler angles (Pitch/Yaw/Roll), 3D coordinate axes from nose, and attention monitoring via `detekface.pose`. | Attention monitoring, driver drowsiness, AR |
| **[Privacy & Anonymization](../examples/privacy_blur.py)** | Real-time identity protection: 8-bit mosaic pixelation, frosted Gaussian blur, and eye redaction bars via `detekface.anonymize`. | GDPR/privacy compliance, public feeds |
| **[Center Stage AI Cameraman](../examples/center_stage.py)** | Cinematic auto-zoom and smooth face tracking framing via `detekface.autocam`. | Video conferencing, smart kiosks, presentation |
| **[Terminal ASCII Streamer](../examples/terminal_face.py)** | Live streaming camera video rendered in 24-bit TrueColor or Matrix green ASCII in console via `detekface.ascii`. | Terminal tools, headless SSH, CLI demos |
| **[Image Detection](image-detection.md)** | Detecting faces in still photos, inspecting coordinates, and integrating with PIL, Matplotlib, and OpenCV. | Batch processing, photo analysis pipelines |
| **[Face Capture & Cropping](face-capture.md)** | Extracting pristine, unannotated face crops from images or live video feeds with smart bounding clamping. | Avatar creation, biometric datasets, profiling |
| **[Configuration & Styling](configuration.md)** | 5 overlay shapes (`rectangle`, `corners`, `circle`, `rounded`, `hud`), color parsing, status banners, and detector thresholds. | UI customization, aesthetic tailoring |
| **[API Reference](api-reference.md)** | Exhaustive reference of every public function, class, dataclass, method, and exception. | Advanced developers, API lookups |
| **[Troubleshooting & FAQ](troubleshooting.md)** | Common issues: Windows 11 camera permissions, missing GUI in headless environments, model download fixes. | Debugging errors, deployment support |
| **[Development & Contributions](development.md)** | Contributor setup, running the test suite (`pytest`), benchmarking, and building release packages. | Open-source contributors |


---

## Recommended Learning Paths

### Path A: Beginners & Quick Experimentation
1. Read **[Installation](installation.md)** to configure your Python environment.
2. Follow **[Quick Start](quickstart.md)** to understand basic computer vision concepts.
3. Launch **[Webcam Streaming](webcam.md)** with a single line of code.

### Path B: Application & Pipeline Integration
1. Review **[Image Detection](image-detection.md)** for file and NumPy array handling.
2. Read **[Face Capture](face-capture.md)** for cropping and saving detected subjects.
3. Customize styling in **[Configuration](configuration.md)** and consult the **[API Reference](api-reference.md)**.

### Path C: Contributors & Maintainers
1. Follow **[Development](development.md)** to clone and install in editable mode (`pip install -e .[dev]`).
2. Run test suites and benchmarks.
3. Check **[Troubleshooting](troubleshooting.md)** for platform-specific considerations.
