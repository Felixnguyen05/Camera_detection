import cv2
import numpy as np


def segment(gray, dark_object=True):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    polarity = cv2.THRESH_BINARY_INV if dark_object else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blur, 0, 255, polarity | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def find_objects(mask, min_area=400, reject_border=True):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    result = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < min_area:
            continue
        if reject_border and (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1):
            continue
        result.append(contour)
    return result


def centroid(contour):
    m = cv2.moments(contour)
    if abs(m['m00']) < 1e-8:
        return None
    return (m['m10'] / m['m00'], m['m01'] / m['m00'])


def normalized_silhouette(contour, size=128, padding=0.76):
    """Normalize scale and translation so rotation search compares shapes fairly."""
    x, y, w, h = cv2.boundingRect(contour)
    canvas = np.zeros((size, size), np.uint8)
    roi_contour = contour.astype(np.float32).copy()
    roi_contour[:, 0, 0] -= x
    roi_contour[:, 0, 1] -= y
    source = np.zeros((h, w), np.uint8)
    cv2.drawContours(source, [roi_contour.astype(np.int32)], -1, 255, -1)
    scale = padding * size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    scaled = cv2.resize(source, (nw, nh), interpolation=cv2.INTER_NEAREST)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = scaled
    return canvas


def rotate_mask(mask, degrees):
    n = mask.shape[0]
    matrix = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), degrees, 1.0)
    return cv2.warpAffine(mask, matrix, (n, n), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class ObjectMatcher:
    def __init__(self, template_file, dark_object=True, min_area=400,
                 shape_threshold=0.25, template_threshold=0.75, angle_step=5):
        gray = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f'Cannot open template: {template_file}')
        self.dark_object = dark_object
        self.min_area = min_area
        self.shape_threshold = shape_threshold
        self.template_threshold = template_threshold
        mask = segment(gray, dark_object)
        candidates = find_objects(mask, min_area=30, reject_border=False)
        if not candidates:
            raise ValueError('No foreground contour in template. Check polarity/background.')
        self.reference = max(candidates, key=cv2.contourArea)
        self.template_area = cv2.contourArea(self.reference)
        base = normalized_silhouette(self.reference)
        self.rotated_templates = [
            (angle, rotate_mask(base, angle)) for angle in range(0, 360, angle_step)
        ]

    def detect(self, gray, method='shape'):
        mask = segment(gray, self.dark_object)
        candidates = find_objects(mask, self.min_area)
        detections = []
        for contour in candidates:
            center = centroid(contour)
            if center is None:
                continue
            distance = float(cv2.matchShapes(self.reference, contour,
                                              cv2.CONTOURS_MATCH_I1, 0.0))
            if not np.isfinite(distance):
                continue
            # Use the same geometry-based rotation estimator in both modes.
            candidate = normalized_silhouette(contour)
            best_angle, best_score = 0, -1.0
            for angle, rotated in self.rotated_templates:
                score = float(cv2.matchTemplate(candidate, rotated,
                                                 cv2.TM_CCORR_NORMED)[0, 0])
                if np.isfinite(score) and score > best_score:
                    best_angle, best_score = angle, score

            if method == 'shape' and distance > self.shape_threshold:
                continue
            if method == 'template' and best_score < self.template_threshold:
                continue
            detections.append({
                'center': center,
                'angle_deg': best_angle,  # rotation relative to reference silhouette
                'shape_distance': distance,
                'template_score': best_score,
                'bbox': cv2.boundingRect(contour),
                'contour': contour,
            })
        detections.sort(key=lambda d: d['shape_distance'] if method == 'shape'
                        else -d['template_score'])
        return detections, mask
