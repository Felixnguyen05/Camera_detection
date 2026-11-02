"""Marker-based component identification and 2D pixel pose.

Conventions: the printed marker's +X goes toward its right edge; +Y toward
its bottom edge. Offsets are in mm measured in the marker plane from its center.
Image angle uses +X toward image right and +Y downward (clockwise positive).
Does not infer true 3D robot pose or detect untagged components.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np


def to_mono8(frame):
    """Normalize a camera/OpenCV frame to contiguous 2-D uint8 grayscale.

    Accepts Mono8 ``(H, W)`` and ``(H, W, 1)`` or BGR ``(H, W, 3)``.
    Raises ValueError for unsupported dimensionality or dtype.
    The ArUco detector expects a grayscale image."""
    a = np.asarray(frame)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    elif a.ndim == 3 and a.shape[2] == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if a.ndim != 2 or a.dtype != np.uint8:
        raise ValueError(f"Expected Mono8 uint8, got {a.shape} {a.dtype}")
    return np.ascontiguousarray(a)


def read_registry(path):
    """Load static component definitions and validate essential fields.

    Args:
        path: Path to ``config/components.json``.
    Returns:
        (config, components_by_marker_id) so each detected marker can be
        mapped to a component in constant time.

    Registry fields describe *fixed* identity/geometry; live positions
    and orientations are calculated from the image, never typed here."""
    path = Path(path)
    cfg = json.loads(path.read_text(encoding='utf-8'))
    dictionary = cfg.get('dictionary', 'DICT_4X4_50')
    if not hasattr(cv2.aruco, dictionary):
        raise ValueError(f"Unknown ArUco dictionary {dictionary}")
    size = float(cfg['marker_size_mm'])
    if size <= 0:
        raise ValueError('marker_size_mm must be positive')
    # Pre-index components by marker ID for fast lookups on each frame.
    entries = {}
    for item in cfg['components']:
        mid = int(item['marker_id'])
        if mid in entries:
            raise ValueError(f'Duplicate marker_id: {mid}')
        if len(item['marker_to_component_mm']) != 2 or len(item['footprint_mm']) != 2:
            raise ValueError('Offsets and footprint must have two elements')
        if min(item['footprint_mm']) <= 0:
            raise ValueError('Footprint dimensions must be positive')
        entries[mid] = item
    return cfg, entries


def projected_points(homography, coords_mm):
    """Project 2-D points from marker-plane millimeters into image pixels.

    Args:
        homography: 3x3 perspective transform mapping marker plane to image.
        coords_mm: N pairs of (x, y), relative to marker center.
    Returns:
        A (N, 2) array of projected image pixel coordinates.

    This assumes the projected points are coplanar with the printed marker."""
    pts = np.asarray(coords_mm, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, homography).reshape(-1, 2)


def component_from_corners(marker_id, observed_corners, item, marker_size_mm):
    """Estimate one registered component's 2-D image pose from its marker.

    Args:
        marker_id: Decoded ID from ArUco.
        observed_corners: Four ordered marker corners in image pixels.
        item: Component registry entry (name, offset, angle, footprint).
        marker_size_mm: Measured printed black-square side length in mm.
    Returns:
        A JSON-serializable pose dict or None if corners are unusable.

    The homography comes from four *known marker corners*; unlike the
    previous SIFT pipeline, no feature matching or RANSAC is needed.
    The reported angle is in image coordinates (+Y downward). This is
    not a calibrated 3-D pose or the outline of a segmented object."""
    corners = np.asarray(observed_corners, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(corners).all():
        return None
    sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    # Reject very small markers: corner noise becomes a large pose error.
    if np.min(sides) < 12:  # pixels, adjust for distance/focus
        return None
    half = marker_size_mm / 2
    model = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=np.float32)
    # Four ordered, known marker corners define a planar projective map.
    # This is NOT SIFT feature matching and does not require RANSAC.
    H = cv2.getPerspectiveTransform(model, corners)
    # Static marker-to-component offset (in marker-local millimeters).
    # If zero, the detected component reference point equals marker center.
    dx, dy = map(float, item['marker_to_component_mm'])
    ang = math.radians(float(item.get('component_angle_deg', 0)))
    ux, uy = math.cos(ang), math.sin(ang)
    # Project a reference axis, not just add angles (valid with perspective).
    length = marker_size_mm * 0.6
    a, b = projected_points(H, [[dx, dy], [dx + length * ux, dy + length * uy]])
    if not np.isfinite([*a, *b]).all():
        return None
    # Image +Y points DOWN: positive angles are clockwise on screen.
    theta = math.degrees(math.atan2(float(b[1]-a[1]), float(b[0]-a[0]))) % 360.0
    fw, fh = map(float, item['footprint_mm'])
    # Registry footprint is only a DRAWING of nominal geometry, not segmentation.
    local_rect = np.array([[-fw/2,-fh/2],[fw/2,-fh/2],[fw/2,fh/2],[-fw/2,fh/2]], dtype=np.float32)
    rotation = np.array([[ux, -uy],[uy, ux]], dtype=np.float32)
    world_rect = local_rect @ rotation.T + [dx,dy]
    outline = projected_points(H, world_rect)
    if not np.isfinite(outline).all():
        return None
    return {
        'marker_id': int(marker_id),
        'component_id': str(item['component_id']),
        'class_id': int(item['class_id']),
        'name': str(item['name']),
        'center_px': [round(float(a[0]), 2),round(float(a[1]), 2)],
        'orientation_deg': round(theta, 2),
        'marker_corners_px': corners.round(2).tolist(),
        'footprint_polygon_px': outline.round(2).tolist(),
        'pose_source': 'marker_projection',
        'calibrated_robot_pose': False
    }


class ComponentDetector:
    def __init__(self, registry_path):
        """Load marker definitions and initialize the configured ArUco detector.

        The dictionary in ``components.json`` must match the printed marker
        dictionary; merely sharing a numeric ID is not sufficient."""
        self.config, self.components = read_registry(registry_path)
        self.marker_size_mm = float(self.config['marker_size_mm'])
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.config['dictionary']))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        self.last_raw_corners = []
        self.last_raw_ids = []

    def detect(self, frame):
        """Detect every visible known marker and estimate component poses.

        Args:
            frame: Current camera image, usually a 2-D Mono8 array.
        Returns:
            (detections, unknown_ids). Unknown markers are identified by
            ArUco but are not registered as components. A missing marker
            produces no component detection (no position is guessed)."""
        gray = to_mono8(frame)
        # Detector returns ALL visible IDs and their corners from one image.
        corners, ids, _ = self.detector.detectMarkers(gray)
        detections = []
        unknown = []
        # Debug state: store *every* decoded marker, including unknown IDs.
        self.last_raw_corners = corners if ids is not None else []
        self.last_raw_ids = [int(v) for v in ids.flatten()] if ids is not None else []
        if ids is None:
            return detections, unknown
        for corners_i, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            # Unknown sticker IDs are not assigned to arbitrary component classes.
            if marker_id not in self.components:
                unknown.append(marker_id)
                continue
            result = component_from_corners(marker_id, corners_i, self.components[marker_id], self.marker_size_mm)
            if result is not None:
                detections.append(result)
        return detections, unknown


def draw_detections(frame, detections, unknown_ids=(), show_footprint=True):
    """Render component IDs, poses and reference footprints for review.

    Args:
        frame: Original camera image.
        detections: Pose records from ComponentDetector.detect().
        unknown_ids: IDs not registered in the config.
        show_footprint: Whether to draw the estimated reference rectangle.
    Returns:
        BGR image ready for cv2.imshow / cv2.imwrite.

    The footprint is a registry-based geometric projection, NOT the
    measured contour of the physical component."""
    gray = to_mono8(frame)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for d in detections:
        corners = np.round(d['marker_corners_px']).astype(np.int32)
        cv2.polylines(out, [corners], True, (0, 230, 230), 2)
        if show_footprint:
            footprint = np.round(d['footprint_polygon_px']).astype(np.int32)
            cv2.polylines(out, [footprint], True, (0, 220, 0), 2)
        x, y = d['center_px']
        theta = math.radians(d['orientation_deg'])
        center = (int(round(x)), int(round(y)))
        tip = (int(round(x+65*math.cos(theta))), int(round(y+65*math.sin(theta))))
        cv2.circle(out, center, 5, (0, 0, 255), -1)
        cv2.arrowedLine(out, center, tip, (255, 0, 255), 2, tipLength=0.2)
        label = f"{d['name']} [{d['component_id']}] ID:{d['marker_id']} theta:{d['orientation_deg']:.0f}"
        y_label = max(22, int(np.min(corners[:, 1]))-12)
        cv2.putText(out, label, (max(5, int(np.min(corners[:,0]))),y_label), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 2)
    cv2.putText(out, f'Components: {len(detections)}  Unknown markers: {len(unknown_ids)}',
                (12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,240,20),2)
    return out
