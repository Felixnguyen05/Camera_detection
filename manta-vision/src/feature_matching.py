"""Known-part localization using SIFT/ORB + robust similarity (partial affine) transform.

Output polygon is the registered *ROI*, not a pixel-exact object segmentation.
No intensity threshold or color class is used. Coordinates refer to full-res camera pixels.
"""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


def mono8(image):
    """Accept VmbPy Mono8 HxW or HxWx1, optionally BGR images for offline tests."""
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f'Expected uint8 grayscale; got {image.shape=} {image.dtype=}')
    return image


@dataclass
class Detection:
    corners: np.ndarray
    center: tuple
    angle_deg: float  # rotation relative to registered image's horizontal axis
    matches: int
    inliers: int
    inlier_ratio: float
    error_px: float
    scale: float
    algorithm: str


class FeatureMatcher:
    def __init__(self, template_path, algorithm='SIFT', *, min_matches=10,
                 min_inliers=8, min_inlier_ratio=.45, max_error_px=4.0,
                 min_scale=.65, max_scale=1.5, min_spread=.10):
        self.reference = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if self.reference is None:
            raise FileNotFoundError(f'Reference missing: {template_path}')
        self.algorithm = algorithm.upper()
        if self.algorithm == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=2000)
            norm = cv2.NORM_L2
            self.ratio = .76
        elif self.algorithm == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=2600, fastThreshold=12)
            norm = cv2.NORM_HAMMING
            self.ratio = .80
        else:
            raise ValueError('Choose SIFT or ORB')
        self.bf = cv2.BFMatcher(norm, crossCheck=False)
        self.ref_kp, self.ref_des = self.detector.detectAndCompute(self.reference, None)
        if self.ref_des is None or len(self.ref_kp) < min_matches:
            raise ValueError(f'{self.algorithm}: too few keypoints in registered component. Capture a sharper ROI.')
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.max_error_px = max_error_px
        self.min_scale, self.max_scale = min_scale, max_scale
        self.min_spread = min_spread
        h,w = self.reference.shape
        self.ref_corners = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        self.ref_center = np.float32([[w/2,h/2]])

    def detect(self, frame):
        scene = mono8(frame)
        kp, descriptors = self.detector.detectAndCompute(scene, None)
        if descriptors is None or len(kp)<self.min_matches:
            return None
        pairs = self.bf.knnMatch(self.ref_des, descriptors, k=2)
        matches = [p[0] for p in pairs if len(p)==2 and p[0].distance < self.ratio*p[1].distance]
        if len(matches)<self.min_matches:
            return None
        src = np.float32([self.ref_kp[m.queryIdx].pt for m in matches])
        dst = np.float32([kp[m.trainIdx].pt for m in matches])
        # A similarity transform allows translation, rotation and uniform scale,
        # but avoids perspective warping/exploding quadrilateral from homography.
        M, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
            maxIters=3000, confidence=.995, refineIters=15)
        if M is None or mask is None or not np.isfinite(M).all():
            return None
        inside = mask.ravel().astype(bool)
        count = int(inside.sum())
        frac = count/len(matches)
        if count<self.min_inliers or frac<self.min_inlier_ratio:
            return None
        # Geometry checks: inliers must cover part of the object, not only
        # a tiny circular hole/single screw. Thresholds are tunable.
        pts = src[inside]
        h,w = self.reference.shape
        spread_x = float(np.ptp(pts[:,0])) / w
        spread_y = float(np.ptp(pts[:,1])) / h
        if min(spread_x, spread_y) < self.min_spread:
            return None
        a,b = M[0,0],M[1,0]
        scale = float(np.hypot(a,b))
        if not (self.min_scale<=scale<=self.max_scale):
            return None
        corners = cv2.transform(self.ref_corners[None], M)[0]
        center = cv2.transform(self.ref_center[None], M)[0,0]
        if not np.isfinite(corners).all() or not np.isfinite(center).all():
            return None
        scene_h, scene_w = scene.shape
        # Partial views cannot reliably supply an entire polygon.
        margin = max(scene_h,scene_w)*.10
        if np.any(corners[:,0] < -margin) or np.any(corners[:,0] > scene_w+margin) or np.any(corners[:,1] < -margin) or np.any(corners[:,1] > scene_h+margin):
            return None
        predicted = cv2.transform(src[inside][None],M)[0]
        errors = np.linalg.norm(predicted-dst[inside], axis=1)
        error = float(np.mean(errors))
        if error>self.max_error_px:
            return None
        angle = float(np.degrees(np.arctan2(b,a))%360)
        return Detection(corners, tuple(map(float,center)),angle,
                         len(matches),count,frac,error,scale,self.algorithm)
