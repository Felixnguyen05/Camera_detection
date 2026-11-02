# Manta G-125B: ArUco Component Detection PoC

This **marker-based** PoC detects *all visible registered markers* in a frame. Each physical component uses a unique marker ID. A component class can have multiple marker IDs. It does NOT visually recognize components without markers, nor segment their actual physical edges. Green outline is a **nominal footprint projection** from configured size/offset, not a measured contour.

## Installation
Use your existing `.venv` with Vimba X + `vmbpy`. Install only OpenCV with `aruco` if absent (avoid mixing pip OpenCV packages):

```powershell
python -c "import cv2; print(cv2.__version__,hasattr(cv2,'aruco'),hasattr(cv2.aruco,'ArucoDetector'))"
# If missing, uninstall opencv-python/opencv-contrib-python and install a supported opencv-contrib-python
pip install -r requirements.txt
```

## One-time setup

1. Edit `config/components.json`. `marker_size_mm` is the side length of the **printed black marker square**, not the surrounding white border. Use identical dictionary for printing and detection.
2. Run `python src/generate_markers.py`; print each `markers/marker_*.png` at the **proper physical size** (be careful of software autoscaling). The white quiet zone must remain. Validate with ruler.
3. Stick marker on a flat visible surface, do not put it on a curved part or in grasp path.
4. Give every physical part its own `marker_id` and `component_id`; duplicates of the same component **class** should still have unique marker IDs. Set `marker_to_component_mm` = `[dx, dy]` from printed marker center to desired component reference/grasp point, **in the marker coordinate system** (+X to marker-right, +Y toward marker-bottom). Set `component_angle_deg` = component's intended axis relative to marker +X, positive clockwise in top-view image. Set `footprint_mm` = nominal width/height in component-local axes. These values are **manually measured/registered once**, never inferred automatically.

## Run

```powershell
python src/discover.py
python src/main.py
```

Close Vimba X Viewer and any other script holding Full Access first. `--camera-id` is optional; default auto-discovers first available camera. With multiple cameras explicitly provide an ID. `S` saves annotated frame and results JSON, `Q` quits and stops streaming.

Offline testing:

```powershell
python -m unittest discover -s tests -v
python src/main.py --image output/your_image.png
```

## Output

`component_id`, `class_id`, `marker_id`, `name`, `center_px`, `orientation_deg`, `footprint_polygon_px`. `center_px` is the saved component reference offset projected through the marker plane. `orientation_deg` is in image coordinates (clockwise from image right). If the marker is directly at the desired reference center, leave `[0,0]`. `pose_source=marker_projection` / `calibrated_robot_pose=false` signal **NOT yet robot coordinates**.

## Accuracy and safety limitations

- Marker identity is a **registered identity**, not independent verification of the underlying physical part. Wrong label or swapped parts => wrong name.
- A hidden/unreadable marker => no detection; never send pose commands when detection absent.
- A visible marker gives its own corners and ID; true 3D pose requires camera intrinsics + distortion calibration, physical marker size, PnP, and hand-eye calibration if camera mounted on robot. Pixel X/Y is **not** robot X/Y.
- Projecting component offsets assumes marker and component reference are in the same *physical plane*. If marker is raised relative to grasp point or component is tilted, this 2D projection becomes approximate. 3D rigid transforms and calibration are needed for real robot pose.
- Even perfect marker identification does not guarantee safe grasp: validate measured offset, work envelope, grasp geometry, stability across multiple frames, and camera-to-robot calibration before robot motion.
- The marker detector internally uses grayscale / adaptive thresholding to decode the marker contrast. This is not the old "component must be brighter than background" segmentation, but markers still need decent illumination, focus, border and adequate pixel size.
- Camera receiving ~30 FPS does not guarantee ArUco processing ~30 FPS; measure on your hardware.

## Research goals

Evaluate multiple marker IDs/classes, rotations 0/30/60/90/180, changes in positions, blur/exposure, distance, marker occlusion, mistaken placements. Metrics include ID correctness, detection rate, 2D reference-point error, orientation error, runtime. For real mm position error add calibration and independently measured ground-truth points.
