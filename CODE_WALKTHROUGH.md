# Code Walkthrough — Manta G-125B ArUco Component Detection

This document explains the functions, runtime flow and data contracts of the source code. All Python functions and class methods also contain English docstrings; the difficult calculations are annotated inline.

## 1. End-to-end runtime

```text
python src/main.py
    |
    +--> ComponentDetector(registry_path)
    |       +--> read_registry()                 [STATIC component metadata]
    |       +--> initialize ArUco dictionary
    |
    +--> MantaCamera.__enter__()                [live camera only]
    |       +--> configure(exposure_us)
    |       +--> cam.start_streaming(LatestFrameStream.callback)
    |
    +--> LatestFrameStream.get()                [newest Mono8 frame]
    |       +--> process(frame)
    |              +--> ComponentDetector.detect()
    |              |       +--> to_mono8()
    |              |       +--> cv2.aruco.ArucoDetector.detectMarkers()
    |              |       +--> component_from_corners() per registered ID
    |              |                 +--> cv2.getPerspectiveTransform()
    |              |                 +--> projected_points()
    |              +--> draw_detections()
    |
    +--> write_latest_detections()              [DYNAMIC output]
    |       +--> output/detections.json
    +--> cv2.imshow()
```

## 2. `src/main.py` — application lifecycle and output

| Function | Inputs | Output / effect |
| --- | --- | --- |
| `main()` | Command-line flags | Loads registry, runs image test or live loop, handles `S`/`Q`, closes camera. |
| `process(frame)` | Latest Mono8 camera image | `(found, unknown, processing_ms, overlay)`; processes ONE frame. |
| `write_latest_detections(output_dir, detections, unknown_ids, processing_ms)` | Per-frame detections | Atomically replaces `output/detections.json` and returns the JSON payload. |

`--json-interval 0.2` means writing *up to* approximately five times per second **while frames arrive**; detection itself runs on each acquired/latest frame. When no marker is visible, the next processed result contains `detections: []`. If the camera/program stops, the last file remains on disk and may be stale; always validate `timestamp_utc` in a robot-side reader.

## 3. `src/aruco_detection.py` — image understanding and geometry

| Function | Purpose |
| --- | --- |
| `to_mono8(frame)` | Converts Mono8 `(H,W,1)` or BGR to supported `(H,W)` grayscale `uint8`. |
| `read_registry(path)` | Validates and loads fixed metadata (`marker_id`, component identity, dimensions and offsets). |
| `ComponentDetector.__init__(registry_path)` | Builds the OpenCV ArUco detector using the configured dictionary and enables subpixel corner refinement. |
| `ComponentDetector.detect(frame)` | Finds visible markers, maps known IDs, returns lists of detections and unknown marker IDs. |
| `component_from_corners(...)` | Uses a planar homography from the four known marker corners to calculate a component reference location, reference-axis angle and nominal footprint. |
| `projected_points(H, coords_mm)` | Projects marker-local 2D millimeter offsets into image pixel coordinates. |
| `draw_detections(...)` | Produces a BGR overlay with marker box, estimated component position, orientation and footprint. |

### Coordinate frames

- **Marker-local coordinates (mm):** origin at marker center; +X toward its right edge; +Y toward its bottom edge.
- **Image coordinates (pixels):** origin top-left; +X right; +Y down; positive screen angle clockwise.
- **Robot/world coordinates:** NOT available until additional camera/workspace or hand-eye calibration.
- `marker_to_component_mm` is a **fixed geometric offset**, not a dynamic position.
- `component_angle_deg` is a **fixed axis alignment relative to the sticker**, not the observed rotation on a table.
- The returned `center_px` and `orientation_deg` are **computed automatically** for every detection.

### Planar projection limitation

`getPerspectiveTransform` maps points on the *same plane as the marker* to the camera image. If the requested component center/grasp point is at a different height, the planar offset is an approximation. The estimated green footprint is a projected nominal rectangle, **not** a vision-measured object contour.

## 4. `src/streaming.py` — low-latency acquisition

| Function | Purpose |
| --- | --- |
| `LatestFrameStream.__init__()` | Initializes a `Queue(maxsize=1)` and counters. |
| `callback(cam, stream, frame)` | Copies a complete Mono8 image, replaces an old queued image if necessary and returns the VmbPy buffer to the SDK in `finally`. |
| `get(timeout)` | Returns newest image or `None` after timeout. |

The latest-frame queue avoids large backlogs when detection processing is slower than acquisition. Returning SDK buffers is mandatory to prevent acquisition from stopping.

## 5. `src/camera.py` — Vimba X camera wrapper

| Method | Purpose |
| --- | --- |
| `MantaCamera.__init__` | Stores camera ID and handles. |
| `__enter__` | Opens SDK and camera, selects compatible pixel format, preferring Mono8. |
| `configure` | Disables trigger and applies manual exposure / optional gain. |
| `read` | **Legacy synchronous helper** returning BGR; not used by `main.py`'s async loop. |
| `__exit__` | Releases camera and SDK resources, including on exceptions. |

## 6. Auxiliary files

- `src/generate_markers.py`: generates printable markers from configured IDs. **Optional if you already have physical ArUco markers.**
- `src/discover.py`: discovers available cameras; this file is a short executable script, not a collection of functions.
- `tests/test_aruco.py`: five offline tests for Mono8 conversion, multiple marker IDs, spatial offset, 90-degree rotation, and empty scenes.

## 7. Config vs live output

**`config/components.json` (manually registered, fixed):** dictionary, real marker size, marker ID, component name/class, fixed marker-to-component offset, component axis alignment, nominal footprint dimensions.

**`output/detections.json` (automatic, changes over time):** UTC timestamp, number of detections, unknown marker IDs, algorithm processing time, detected `marker_id`, mapped component identity, measured `center_px`, `orientation_deg`, projected corners and footprint.

Do not use image pixel coordinates as calibrated robot millimeters. A system consuming this JSON must check both `robot_pose_calibrated` and the timestamp.

## 8. Commands

From project root on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
python src/main.py --image output/example_three_markers.png
python src/main.py --json-interval 0.2
```

The included sample config still uses `DICT_4X4_50` and IDs 0/1/2. **Configure the dictionary and IDs to match your actual printed markers before live testing.**
