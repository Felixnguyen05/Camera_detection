"""Realtime markerless instance segmentation or one-image offline inference.

Windows examples:
    python src/main.py --image sample/demo_scene.png --weights models/best.pt
    python src/main.py --camera --weights models/best.pt --exposure-us 5000
"""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2

from vision.inference import ComponentSegmenter
from vision.outputs import make_document, safe_write_json, save_masks
from vision.visualize import render

ROOT = Path(__file__).resolve().parents[1]


def process(segmenter, frame, *, source, output, write_masks=False, prefix='snapshot'):
    """Infer image masks, derive geometry, and build a JSON-ready document."""
    detections, masks, ms = segmenter.predict(frame)
    if write_masks:
        save_masks(output / 'masks', masks, detections, prefix)
    doc = make_document(detections, ms, source=source,
                        model_path=str(segmenter.weights), frame_shape=frame.shape)
    return doc, masks, render(frame, detections, masks, ms)


def run(args):
    """Prepare the model then process a still image or live Mono8 Manta frames."""
    if args.json_interval <= 0 or not 0 < args.scale <= 3:
        raise ValueError('json interval must be >0; scale must be between 0 and 3')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    segmenter = ComponentSegmenter(args.weights, confidence=args.confidence,
                                    imgsz=args.imgsz, device=args.device,
                                    preprocess_mode=args.preprocess,
                                    min_area=args.min_area)
    target = output / 'detections.json'
    print('[MODEL]', segmenter.weights)
    print('[OUTPUT]', target)
    if args.image:
        frame = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise FileNotFoundError(args.image)
        doc, masks, overlay = process(segmenter, frame, source=str(args.image),
                                      output=output, write_masks=True, prefix='offline')
        ok = safe_write_json(target, doc)
        cv2.imwrite(str(output / 'offline_overlay.png'), overlay)
        print(f'[OFFLINE] detected={len(doc["detections"])} saved_json={ok}')
        print('[SAVED]', output / 'offline_overlay.png')
        return
    if not args.camera:
        raise ValueError('Choose --image path.png or --camera')
    # Imports deferred: image-only inference requires no Allied Vision SDK.
    from camera import MantaCamera
    from streaming import LatestFrameStream
    latest = LatestFrameStream()
    last_write = 0.0
    last_warning = 0.0
    last_count = None
    print('[CAMERA] Close Vimba X Viewer before starting. Q quits; S saves masks/overlay.')
    with MantaCamera(camera_id=args.camera_id) as camera:
        camera.configure(exposure_us=args.exposure_us)
        camera.cam.start_streaming(handler=latest.callback, buffer_count=5)
        try:
            while True:
                frame = latest.get(timeout=0.5)
                if frame is None:
                    if cv2.waitKey(1) & 255 in (ord('q'), 27):
                        break
                    continue
                doc, masks, overlay = process(segmenter, frame, source='manta_camera',
                                              output=output)
                now = time.monotonic()
                if now - last_write >= args.json_interval:
                    if safe_write_json(target, doc):
                        last_write = now
                        if len(doc['detections']) != last_count:
                            print(f'[JSON OK] detected={len(doc["detections"])} '
                                  f'latency={doc["inference_ms"]:.1f}ms '
                                  f'timestamp={doc["timestamp_utc"]}', flush=True)
                            last_count = len(doc['detections'])
                    elif now - last_warning >= 3:
                        print('[JSON WARN] Could not replace detections.json: check Windows file locks', flush=True)
                        last_warning = now
                shown = cv2.resize(overlay, None, fx=args.scale, fy=args.scale) if args.scale != 1 else overlay
                cv2.imshow('Manta Markerless Instance Segmentation', shown)
                key = cv2.waitKey(1) & 255
                if key in (ord('q'), 27):
                    break
                if key == ord('s'):
                    stamp = time.strftime('%Y%m%d_%H%M%S')
                    # Masks saved only for a snapshot to avoid huge disk growth.
                    save_masks(output / 'masks', masks, doc['detections'], stamp)
                    safe_write_json(output / f'detections_{stamp}.json', doc)
                    cv2.imwrite(str(output / f'overlay_{stamp}.png'), overlay)
                    print('[SAVED] Snapshot', stamp)
        finally:
            try:
                camera.cam.stop_streaming()
            finally:
                cv2.destroyAllWindows()


def cli():
    """Parse CLI options; do not silently use generic COCO model weights."""
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--camera', action='store_true')
    mode.add_argument('--image', type=Path)
    p.add_argument('--weights', type=Path, default=ROOT/'models/best.pt')
    p.add_argument('--output', type=Path, default=ROOT/'output')
    p.add_argument('--camera-id', default=None)
    p.add_argument('--exposure-us', type=float, default=5000)
    p.add_argument('--confidence', type=float, default=.35)
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--device', default='auto', help='auto, cpu, or 0 for CUDA GPU')
    p.add_argument('--preprocess', choices=['none', 'clahe'], default='none')
    p.add_argument('--min-area', type=int, default=80)
    p.add_argument('--json-interval', type=float, default=.3)
    p.add_argument('--scale', type=float, default=.85)
    return p.parse_args()

if __name__ == '__main__':
    try:
        run(cli())
    except (ValueError, FileNotFoundError) as exc:
        print('[ERROR]', exc, file=sys.stderr)
        sys.exit(2)
