# Geometry, orientation, mask limits and integration contract

## What is a mask?
`mask[i, j]=1` identifies a pixel predicted as belonging to one component. Each instance has a **different mask**, including multiple components of the same class. A polygon (green contour) traces its outer boundary, not a rectangle. Masks only represent **visible** pixels; occluded part geometry cannot be truthfully recovered from one view.

## Different types of contours

1. **Outer silhouette:** generated from instance mask with `cv2.RETR_CCOMP` and `CHAIN_APPROX_NONE`. 400 sampled points at most in JSON (for size); full mask saved on snapshot.
2. **Holes:** if a prediction truly has 0-pixels inside the object's positive region, contours for enclosed holes may be extracted. **Standard Ultralytics polygon labels do not encode holes**, so they cannot be guaranteed by the trained model.
3. **Internal structural features:** black speaker ring, screw head, indentation, non-through hole. These may be *inside* the foreground mask; the mask will not automatically separate them. Need per-part models or custom edge/circle detector. Do not claim the single instance mask captures each internal detail.

## Orientation from PCA

We collect all positive mask pixel coordinates (x,y), compute their covariance, find the principal eigenvector and calculate `atan2(dy, dx) % 180`. This is a **line**, not a directed arrow, so 30° and 210° are indistinguishable; emitted `direction_ambiguous=true`. If the shape is rotationally symmetric, eigenvalue ratio ~1 and no stable axis exists: `orientation_deg=null` and `axis_reliable=false`. The threshold `1.2` is tunable.

**Robot integration:** Define part-specific grasp/reference landmarks and correct 0°/180° orientation. A future component-specific template/matching stage can refine identity/direction *inside segmented component ROI*, but it should not replace the mask. Camera intrinsic calibration, work-plane or 3D pose, robot transforms and safety checks remain separate.

## Realism of performance

Inference at `imgsz=960` can be slower than Manta's 30 FPS; the newest-frame stream avoids accumulating stale frames but doesn't guarantee 30 processed FPS. Measure processing time with your actual GPU/CPU, number of instances and model.

Image preprocessing such as CLAHE may help or hurt, test with held-out data; it is not illumination invariance. Keep illumination diffuse, reasonably uniform; metal reflection and motion blur remain challenges. Do not use screenshot images with colored overlays as training data, unless annotated accordingly.

## JSON structure

`output/detections.json` is **latest state** (overwritten) not a persistent result log. Key fields: `detected_count`, `image_size`, `coordinate_system`, `inference_ms`, `robot_pose_calibrated`, and per-instance `name`, `confidence`, `center_px`, `orientation_deg`, `outer_contour_px`, `holes_contours_px`, `axis_reliable`, `direction_ambiguous`.

An instance index is transient for each frame; it is not a physical serial number. `calibrated_robot_pose=false` is intentionally constant: do not feed these image coordinates into robot motion commands.
