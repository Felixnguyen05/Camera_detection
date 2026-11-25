"""Ultralytics model adapter. Model weights must be trained on YOUR components."""
from __future__ import annotations
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from .geometry import gray_to_bgr, mask_geometry, normalize_gray, preprocess


class ComponentSegmenter:
    """Load a trained instance-segmentation model and infer all instances."""
    def __init__(self, weights: str | Path, *, confidence=0.35, imgsz=960,
                 device='auto', preprocess_mode='none', min_area=80):
        from ultralytics import YOLO
        self.weights = Path(weights).resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(f'No trained model: {self.weights}. Run train.py first.')
        self.model = YOLO(str(self.weights))
        if self.model.task != 'segment':
            raise ValueError(f'Weights task must be segment, got {self.model.task}. Use best.pt from segmentation training.')
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self.device = None if device == 'auto' else device
        self.preprocess_mode = preprocess_mode
        self.min_area = int(min_area)

    def predict(self, image: np.ndarray) -> tuple[list[dict], list[np.ndarray], float]:
        """Return per-instance label, precise mask geometry and latency.

        No training or SAM is performed during live inference. Model predicts
        from RGB-replicated Mono8 so camera is still acquired in Mono8.
        """
        gray = preprocess(normalize_gray(image), self.preprocess_mode)
        bgr = gray_to_bgr(gray)
        start = perf_counter()
        args = dict(source=bgr, conf=self.confidence, imgsz=self.imgsz,
                    retina_masks=True, verbose=False)
        if self.device is not None:
            args['device'] = self.device
        result = self.model.predict(**args)[0]
        elapsed_ms = (perf_counter() - start) * 1000
        dets = []
        masks = []
        if result.boxes is None or result.masks is None:
            return dets, masks, elapsed_ms
        raw_masks = result.masks.data.cpu().numpy()
        boxes = result.boxes
        n = min(len(boxes), len(raw_masks))
        names = result.names
        for i in range(n):
            full = raw_masks[i]
            if full.shape[:2] != gray.shape[:2]:
                # No bilinear resampling: would distort thin tabs/hole edges.
                full = cv2.resize(full.astype(np.float32), (gray.shape[1], gray.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            binary = np.uint8(full > 0.5)
            geo = mask_geometry(binary, min_pixels=self.min_area)
            if geo is None:
                continue
            class_id = int(boxes.cls[i].item())
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            dets.append({
                'instance_index': len(dets),
                'class_id': class_id,
                'name': str(class_name),
                'confidence': round(float(boxes.conf[i].item()), 4),
                'detection_method': 'yolo_instance_segmentation',
                **geo,
                'calibrated_robot_pose': False,
            })
            masks.append(binary)
        # Do not pretend a per-frame index is a persistent physical component ID.
        return dets, masks, elapsed_ms
