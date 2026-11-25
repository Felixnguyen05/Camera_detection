"""Reliable JSON output and optional segmentation mask artifacts on Windows."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid
import cv2
import numpy as np


def utc_now():
    """Machine-readable timestamp for consumers to reject stale detections."""
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def make_document(detections: list[dict], inference_ms: float, *,
                  source: str, model_path: str, frame_shape: tuple,
                  acquisition_age_ms: float | None = None) -> dict:
    """Build same-schema output even for empty frames or missing detections."""
    h, w = frame_shape[:2]
    return {
        'timestamp_utc': utc_now(),
        'source': source,
        'model_path': str(model_path),
        'detected_count': len(detections),
        'inference_ms': round(float(inference_ms), 2),
        'acquisition_age_ms': None if acquisition_age_ms is None else round(float(acquisition_age_ms), 2),
        'image_size': {'width': int(w), 'height': int(h)},
        'coordinate_system': 'image_pixels',
        'robot_pose_calibrated': False,
        'detections': detections,
    }


def safe_write_json(path: Path, data: dict, *, retries=5, delay=0.06) -> bool:
    """Write then replace using unique temporary file; survive Windows file locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        for attempt in range(retries):
            try:
                os.replace(tmp, path)
                return True
            except PermissionError:
                if attempt == retries - 1:
                    return False
                time.sleep(delay)
        return False
    except OSError:
        return False
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def save_masks(folder: Path, masks: list[np.ndarray], detections: list[dict], prefix: str):
    """Save lossless full-resolution mask PNGs for independent outline validation."""
    folder.mkdir(parents=True, exist_ok=True)
    for i, (mask, det) in enumerate(zip(masks, detections)):
        name = f'{prefix}_{i:03d}_class{det["class_id"]}.png'
        cv2.imwrite(str(folder / name), np.uint8(mask > 0) * 255)
        det['mask_path'] = (Path('masks') / name).as_posix()
