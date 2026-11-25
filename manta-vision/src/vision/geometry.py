"""Derive image geometry from actual instance masks, NOT bounding rectangles.

The PCA principal axis is intrinsically unoriented (0 vs 180 degrees). A robot
cannot use this value as a directed grasp angle without additional landmarks.
"""
from __future__ import annotations
import cv2
import numpy as np


def normalize_gray(image: np.ndarray) -> np.ndarray:
    """Normalize Mono8 / singleton Mono8 / BGR frames to 2-D uint8."""
    if image is None:
        raise ValueError('Image is None')
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]
    if image.ndim == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f'Expected grayscale or 3-channel image; got {image.shape}')
    if image.dtype != np.uint8:
        raise ValueError(f'Expected uint8 image; got {image.dtype}')
    return image


def gray_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert camera's Mono8 frame for models and colored overlays."""
    return cv2.cvtColor(normalize_gray(image), cv2.COLOR_GRAY2BGR)


def preprocess(image: np.ndarray, mode: str = 'none') -> np.ndarray:
    """Optional contrast normalization; verify changes against held-out data."""
    gray = normalize_gray(image)
    if mode == 'none':
        return gray
    if mode == 'clahe':
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    raise ValueError(f'Unsupported preprocessing mode: {mode}')


def mask_geometry(mask: np.ndarray, *, min_pixels: int = 80,
                  contour_limit: int = 400) -> dict | None:
    """Calculate centroid, undirected PCA axis, outer contour and internal holes.

    Args:
        mask: One instance's full-resolution binary or probability mask (H,W).
        min_pixels: Reject tiny, unstable segments.
        contour_limit: Maximum contour points in JSON; mask PNG retains pixels.

    Returns:
        JSON-compatible dict; None when mask empty/too small. `orientation_deg`
        is in [0,180), clockwise in image coordinates; `direction_ambiguous` is
        always true for pure PCA. Outer contour follows the segmented silhouette.
        Internal holes only exist when the mask itself excludes these pixels.
    """
    fg = np.asarray(mask) > 0.5
    if fg.ndim != 2:
        raise ValueError('Expected a 2-D single-instance mask')
    area = int(fg.sum())
    if area < min_pixels:
        return None
    binary = np.ascontiguousarray(fg.astype(np.uint8) * 255)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    # Dominant connected region represents the detected instance.
    outer_idx = max(range(len(contours)), key=lambda k: cv2.contourArea(contours[k]))
    outer = contours[outer_idx][:, 0, :]
    if len(outer) < 3:
        return None
    # Use all foreground pixels, including narrow mounting tabs, for centroid/PCA.
    yy, xx = np.nonzero(fg)
    x = xx.astype(np.float64)
    y = yy.astype(np.float64)
    center_x, center_y = float(x.mean()), float(y.mean())
    samples = np.column_stack((x - center_x, y - center_y))
    covariance = samples.T @ samples / float(area)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    major = eigvecs[:, int(np.argmax(eigvals))]
    angle = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    major_val = float(eigvals.max())
    minor_val = float(eigvals.min())
    ratio = major_val / max(minor_val, 1e-9)
    # Near circular/square shapes have no reliable principal orientation.
    axis_reliable = bool(ratio >= 1.20)
    if not axis_reliable:
        angle_out = None
    else:
        angle_out = round(angle, 2)
    holes = []
    # Hierarchy: next, previous, first_child, parent. Children of main contour are holes.
    if hierarchy is not None:
        h = hierarchy[0]
        for i, c in enumerate(contours):
            if int(h[i][3]) == outer_idx and cv2.contourArea(c) >= 6:
                holes.append(_compact_points(c[:, 0, :], contour_limit))
    ys, xs = np.where(fg)
    return {
        'center_px': [round(center_x, 2), round(center_y, 2)],
        'orientation_deg': angle_out,
        'orientation_method': 'pca_undirected',
        'direction_ambiguous': True,
        'axis_reliable': axis_reliable,
        'axis_variance_ratio': round(float(ratio), 3),
        'area_px': area,
        'bbox_xyxy_px': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        'outer_contour_px': _compact_points(outer, contour_limit),
        'holes_contours_px': holes,
    }


def _compact_points(points: np.ndarray, max_points: int) -> list[list[int]]:
    """Reduce JSON size without affecting the full-resolution mask export."""
    pts = np.asarray(points).reshape(-1, 2)
    if len(pts) > max_points:
        indices = np.linspace(0, len(pts) - 1, max_points, dtype=np.int64)
        pts = pts[indices]
    return [[int(a), int(b)] for a, b in pts]


def polygons_to_line(contours: list[list[int]], height: int, width: int, class_id: int) -> str:
    """Format one outer polygon as Ultralytics normalized segmentation label."""
    p = np.asarray(contours, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3 or width <= 0 or height <= 0:
        raise ValueError('Polygon requires >=3 vertices and positive image size')
    p[:, 0] = np.clip(p[:, 0] / width, 0, 1)
    p[:, 1] = np.clip(p[:, 1] / height, 0, 1)
    return str(int(class_id)) + ' ' + ' '.join(f'{v:.6f}' for v in p.flatten())
