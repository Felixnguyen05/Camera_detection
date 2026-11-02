# SOURCE REFERENCES / CODE PROVENANCE

## What this project is
This Manta ArUco PoC was assembled and authored for the user's machine-vision project , building on camera acquisition work already developed in the user's project. It is **not** a verbatim clone of a third-party GitHub repository. Library APIs and standard algorithms are credited below. This is not a claim that a particular external repository supplied the exact text of each project file.

## Official references

1. Allied Vision Vimba X / VmbPy Python API Manual — https://docs.alliedvision.com/vimba_x/pythonAPIManual.html
2. Allied Vision VmbPy official code/examples — https://github.com/alliedvision/VmbPy
3. Allied Vision SDK examples — https://docs.alliedvision.com/examplesOverview.html
4. OpenCV ArUco Detection tutorial — https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
5. OpenCV ArUco API reference — https://docs.opencv.org/4.x/d9/d6a/group__aruco.html
6. OpenCV geometric image transformations (getPerspectiveTransform/perspectiveTransform) — https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
7. OpenCV feature2D matchers and homography — https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html and https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
8. OpenCV affine estimation — https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
9. OpenCV camera calibration / PnP — https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
10. NumPy — https://numpy.org/doc/stable/
11. Python queue — https://docs.python.org/3/library/queue.html
12. Python JSON — https://docs.python.org/3/library/json.html
13. Python os.replace — https://docs.python.org/3/library/os.html#os.replace
14. Python argparse — https://docs.python.org/3/library/argparse.html
15. Python unittest — https://docs.python.org/3/library/unittest.html

## File → provenance mapping
- src/camera.py: User's prior Manta camera PoC workflow and Allied Vision VmbPy public API.
- src/streaming.py: User's prior async camera stream workflow; VmbPy callback/buffer API; Python queue patterns.
- src/aruco_detection.py: Project-specific ID registry and pose logic implemented using standard OpenCV ArUco and projective transformations.
- src/main.py: Project-specific CLI/UI/persistence integration using OpenCV, JSON, Python stdlib.
- src/generate_markers.py: OpenCV aruco marker generation API.
- src/discover.py: VmbPy device enumeration API.
- config/components.json: Example user-defined registry, not sourced externally. Replace ID/dictionary/size with measured values.
- tests/test_aruco.py: Project-specific tests using OpenCV and Python unittest.

## Important limitations
The green quadrilateral visualizes a nominal registered footprint, not segmentation of the physical object. Current coordinates are camera image pixels, not calibrated robot coordinates. Marker identity relies on a correct registry, tag placement, and unobstructed, readable fiducials.

## Important setup
The bundled `config/components.json` is a sample (`DICT_4X4_50`); the user later identified markers read with `DICT_ARUCO_ORIGINAL` and IDs such as 277, 628, 896. Do NOT overwrite your working local registry when copying updates. If working with the user's printed marker set, configure dictionary and verified IDs locally and measure actual marker size/offsets.

## Files represented
This copy combines the Fully Commented base, ArUco debug overlay patch, and latest JSON sync/debug main.py patch. It is provided as a provenance/reading bundle rather than as a certified hardware deployment build.
