"""Draw actual mask edges rather than rotated rectangles."""
import cv2
import numpy as np
from .geometry import gray_to_bgr


def render(image, detections, masks, inference_ms=0):
    """Overlay semitransparent masks, outer borders, centroid, PCA axis and class."""
    out = gray_to_bgr(image)
    palette = [(48, 210, 70), (255, 180, 60), (200, 80, 255), (80, 230, 240)]
    for i, (d, m) in enumerate(zip(detections, masks)):
        color = palette[d['class_id'] % len(palette)]
        fg = m.astype(bool)
        if np.any(fg):
            overlay = np.array(color, dtype=np.uint8)
            out[fg] = (out[fg].astype(np.float32) * .76 + overlay.astype(np.float32) * .24).astype(np.uint8)
        outer = np.array(d['outer_contour_px'], dtype=np.int32).reshape(-1, 1, 2)
        if len(outer) >= 3:
            cv2.polylines(out, [outer], True, color, 2, cv2.LINE_AA)
        for hole in d['holes_contours_px']:
            if len(hole) > 2:
                cv2.polylines(out, [np.array(hole, dtype=np.int32)], True, (255, 255, 0), 1)
        x, y = map(int, map(round, d['center_px']))
        cv2.circle(out, (x, y), 5, (0, 0, 255), -1)
        theta = d.get('orientation_deg')
        if theta is not None:
            # PCA axis has 180-degree ambiguity, show a two-ended line.
            a = np.deg2rad(theta)
            vx, vy = 55 * np.cos(a), 55 * np.sin(a)
            start = (int(x - vx), int(y - vy))
            end = (int(x + vx), int(y + vy))
            cv2.line(out, start, end, (255, 0, 255), 2, cv2.LINE_AA)
        label = f"{d['name']} #{i+1} {d['confidence']:.2f} angle={theta if theta is not None else 'N/A'}"
        bx = max(4, int(d['bbox_xyxy_px'][0]))
        by = max(48, int(d['bbox_xyxy_px'][1]) - 8)
        cv2.putText(out, label, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
    cv2.putText(out, f'Instances: {len(detections)} | YOLO {inference_ms:.1f} ms | Q quit | S snapshot',
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (40, 255, 40), 2, cv2.LINE_AA)
    return out
