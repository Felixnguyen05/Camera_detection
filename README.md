# Manta Markerless Multi-Component Instance Segmentation

**Production-minded research starter for Allied Vision Manta G-125B / Vimba X / OpenCV / Ultralytics YOLO-seg.**

## Truth-in-advertising / current status

- **Implemented and CPU-tested here:** image normalization; multi-instance *mask processing*; full outer contours (not rectangles); optional hole contours; centroid; PCA undirected orientation; JSON export; Windows-safe JSON writes; synthetic example; label tooling; model training and validation scripts.
- **Requires your real data:** a trained `models/best.pt`. Generic COCO weights **cannot be assumed to recognize your industrial speaker and electronic module.** No trained weights or real annotations are included. Real-camera/inference integration requires Windows testing.
- **Not yet implemented:** camera intrinsics/robot calibration, 3-D pose, directed orientation, grasp point, tracking/persistent physical instance ID, real hardware benchmark, pixel-perfect internal part segmentation.
- **Image illumination:** cannot be made wholly independent of lighting. Training with diverse exposure/lighting and good diffuse lighting matter.

## A. Windows one-time installation

Open `manta-vision` or another copy of this new project in VS Code. **Do not overlay this entire new project on your working ArUco project.** Save them separately (for example `manta-segmentation`).

```powershell
cd "C:\PATH\TO\Manta_Markerless_Segmentation"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import cv2, torch, ultralytics; print(cv2.__version__, torch.__version__, ultralytics.__version__)"
```

If you have NVIDIA CUDA and want GPU training, install the **correct** PyTorch/CUDA version from [PyTorch official installer](https://pytorch.org/get-started/locally/) before `pip install -r requirements.txt`; verify `torch.cuda.is_available()`.

`vmbpy` comes with **Allied Vision Vimba X** and its corresponding Python bindings, not `pip install vmbpy`. Test `python -c "import vmbpy; print('VmbPy OK')"`. If missing, follow the installed Vimba X SDK's Python setup and use its supported Python version. If Vimba X on your computer only supports another Python version, create a compatible virtual environment. Ensure Vimba X Viewer is CLOSED while running Manta scripts.

Important: use only **one** OpenCV package in the environment (`opencv-python` OR `opencv-contrib-python`). If this venv was copied from the old ArUco project and contains contrib, prefer keeping that instead of installing both.

## B. Before training: prove geometry and I/O (no model/camera needed)

```powershell
python -m unittest discover -s tests -v
python src/demo_geometry.py
```

Open `output/synthetic_demo/preview.png` and `detections.json`. It shows two artificial tabbed shapes with traced contours, center points, and **undirected** PCA axes. **It is not YOLO recognition evidence.**

## C. Capture images from Manta G-125B

First create distinct acquisition sessions for train and validation; do NOT take adjacent video frames into train and val randomly because nearly identical frames inflate results.

```powershell
python src/capture_dataset.py --split train --session trainDay1 --exposure-us 5000
python src/capture_dataset.py --split val --session valDay2 --exposure-us 5000
```

Controls: **S** capture current frame, **A** toggle auto capture at interval (default 1.5 s), **Q** quit.

Collect different part poses: 0–360° rotations, both part classes, 1–5 simultaneous components, variation in light/background/camera distance, empty scenes, occlusions. *Capture all instances, not just the most obvious one.* Keep each image in original 1292×964 Mono8 PNG resolution. Start around 150–300 diverse images (not a guarantee of accuracy). Hold out different sessions for validation.

If capture is inconvenient, copy your existing raw camera images to `dataset/images/train` and `dataset/images/val`; use distinct session filename prefixes, e.g. `trainA_...` versus `valB_...`.

## D. Manually annotate full outlines

```powershell
python src/annotate.py --split train
python src/annotate.py --split val
```

**Left mouse:** add a vertex along the *whole external part silhouette*, including metal mounting tabs. **Right mouse:** undo a vertex. **Enter:** complete a polygon for ONE component. **1:** Speaker (`class_id=0`); **2:** Electronic Module (`class_id=1`). Repeat for multiple components in the SAME image. **S:** save labels for image; **N:** next; **P:** previous; **U:** remove last completed polygon; **Q:** quit. `N` does NOT auto-save; save explicitly before moving to the next image. For empty scenes press S with zero polygons (produces an empty text file).

The label at `dataset/labels/train/photo.txt` matches `dataset/images/train/photo.png`. Format: `<class_id> x1 y1 x2 y2 ...` normalized 0..1.

**IMPORTANT**: Training labels are *external instance outlines*. YOLO polygon format does not support internal holes natively. Thus the visible speaker ring/screw holes are **not automatically learned as separate labelled parts**; their contour must be handled via supplementary part annotations or specialized geometry/feature logic. The pipeline extracts holes only if the predicted binary mask actually contains them. See `docs/GEOMETRY_AND_LIMITATIONS.md`.

### Optional assisted labeling with SAM 2 (requires SAM weights and more resources)

```powershell
python src/sam_assist.py --image dataset/images/train/trainDay1_....png --class-id 0
```

Select a rectangle tightly containing **one** component and press Enter. SAM2 generates a proposed mask. Press **Y** to append its outer contour to the corresponding segmentation label *only after visually reviewing it*; **N** rejects. Each repetition adds ONE instance. You can then use `annotate.py` to review/replace labels. The first run may download `sam2.1_t.pt` from Ultralytics; some configurations require GPU and may take time. This assisted mode is not required and is not used during realtime detection.

**If the ROI selection window is scaled by your desktop OS, carefully verify point locations in labels.**

## E. Audit and train

```powershell
python src/audit_dataset.py
python src/train.py --epochs 80 --imgsz 960 --batch 4 --device auto
```

Defaults: transfer learning from `yolo26n-seg.pt`; `device=auto` selects GPU if available, otherwise CPU. Windows uses `workers=0` by default for reliability. Training downloads public pretrained weights on first use (internet required). Requires a train/val dataset with all images annotated. The script writes `dataset/data.yaml` with the **absolute** path, then saves `models/best.pt` after training. `runs/components_seg*/` contains losses and metrics.

If CPU-only and RAM limited, begin with:

```powershell
python src/train.py --epochs 40 --imgsz 640 --batch 2 --device cpu
```

To improve boundaries, try higher `imgsz=960/1280`, more diverse data, accurate masks, and a larger model **only after** checking validation performance.

## F. Test one real image BEFORE starting camera

```powershell
python src/main.py --image dataset/images/val/valDay2_....png --weights models/best.pt --imgsz 960
```

Inspect `output/offline_overlay.png`, `output/detections.json`, and `output/masks/`. Masks are saved as **actual per-pixel PNGs**, so the outlines can be inspected beyond polygon point sampling.

## G. Live Manta multi-component inference

Close Vimba X Viewer, then:

```powershell
python src/main.py --camera --weights models/best.pt --exposure-us 5000 --imgsz 960 --confidence 0.35 --scale 0.85
```

Control **S** = timestamped overlay + JSON + full-resolution masks; **Q/ESC** = exit. The live `output/detections.json` refreshes every 0.3 s by default (`--json-interval 0.3`), and resets to `detections: []` if no components are found. If Windows temporarily locks the JSON file, writes fail with a warning, **without crashing the camera**; consuming robot software must reject stale timestamps.

**All estimated centers/angles are IMAGE coordinates, not calibrated robot coordinates**. `calibrated_robot_pose=false` always. `orientation_deg` is PCA's undirected axis in `[0,180)`; it is null when the object shape is near circular. It is NOT an end-effector rotation for pick-and-place without a part-specific landmark/reference and robot calibration. Multiple detections per class use `instance_index`, which is **per frame**, NOT persistent physical identity. For unique identity, add multi-object tracking and data association or markers.

## H. Evaluate segmentation quality

```powershell
python src/evaluate.py --weights models/best.pt --imgsz 960
```

Writes `output/validation_metrics.json` including class-aware instance recall/precision, average IoU and boundary F1 of matched masks, and inference latency. Ultralytics training also supplies segmentation mAP. Important: the IoU/boundary mean is averaged over matched objects, whereas missed objects are reflected in precision/recall; do not report matched IoU alone as overall system performance. True 2D pose accuracy requires labeled reference pose/ground truth, which this starter project does not invent.

## File map

```
src/camera.py                Existing Manta driver adapted from prior ArUco PoC
src/streaming.py             Existing Mono8 newest-frame VmbPy callback
src/capture_dataset.py      Capture train/validation camera images
src/annotate.py             Manual per-instance polygon labeling
src/sam_assist.py            Optional SAM 2 assisted polygons
src/audit_dataset.py        Validate split/labels/class IDs
src/train.py                Transfer-learning segmentation
src/inference.py            (Actual location src/vision/inference.py)
src/vision/geometry.py      Full outer contour, holes, PCA, center
src/vision/inference.py     Segmentation mask prediction and geometry
src/vision/visualize.py     Draw mask boundaries & per-instance values
src/vision/outputs.py       Atomic-ish Windows JSON with retry, mask PNGs
src/main.py                 Image or camera inference, snapshots
src/evaluate.py             Held-out mask IoU and boundary scoring
src/demo_geometry.py        Honest synthetic smoke test
config/classes.json         Class names and orientation assumptions
```

## Authoritative references

- [Ultralytics YOLO instance segmentation](https://docs.ultralytics.com/tasks/segment/)
- [Ultralytics segmentation dataset format](https://docs.ultralytics.com/datasets/segment/)
- [Ultralytics SAM2 prompt examples](https://docs.ultralytics.com/models/sam-2/)
- [Ultralytics original-resolution masks: retina_masks](https://docs.ultralytics.com/modes/predict/)
- [Allied Vision Vimba X Python API](https://docs.alliedvision.com/vimba_x/pythonAPIManual.html)
- [VmbPy camera streaming API](https://www.alliedvisiontech.com/api/VmbPy/Camera.html)
- [OpenCV contours/hierarchy](https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html)
- [OpenCV moments](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
- [Ultralytics licensing](https://www.ultralytics.com/license)

**Licensing:** Ultralytics provides AGPL-3.0 and Enterprise licensing options. Check obligations before deploying proprietary/commercial code. VmbPy/SDK has separate Allied Vision conditions.
