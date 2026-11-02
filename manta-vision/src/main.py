"""Live multi-component ArUco registration-based localization.
Run from project root: python src/main.py
"""
import argparse
import json
import time
import os
import uuid
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import cv2
from aruco_detection import ComponentDetector, draw_detections

ROOT=Path(__file__).resolve().parents[1]


def write_latest_detections(output_dir, detections, unknown_ids, processing_ms):
    """Persist the latest detection state to ``output/detections.json``.

    Args:
        output_dir: Folder where the JSON file is stored.
        detections: List of registered component pose dictionaries.
        unknown_ids: Detected ArUco IDs missing from the registry.
        processing_ms: Detector runtime for the current frame (milliseconds).

    Returns:
        JSON-compatible dictionary written to disk.

    Important:
        Writes a temporary file and atomically replaces the target. The
        previous JSON is replaced; this is NOT a history log. A stopped
        program cannot update the timestamp, so consumers must reject stale
        results before using them to control a robot."""
    # Dynamic fields come from the CURRENT frame, not from components.json.
    document = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
        'detected_count': len(detections),
        'unknown_marker_ids': [int(mid) for mid in unknown_ids],
        'processing_ms': round(float(processing_ms), 2),
        'coordinate_system': 'image_pixels',
        'robot_pose_calibrated': False,
        'detections': detections,
    }
    target = output_dir / 'detections.json'
    # A unique temporary file avoids conflicts with an earlier failed write.
    tmp = output_dir / f'detections.{uuid.uuid4().hex}.tmp'
    try:
        tmp.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
        # On Windows, readers or antivirus can briefly lock the destination.
        # Retry a few times; never terminate camera acquisition for an I/O error.
        for attempt in range(5):
            try:
                os.replace(tmp, target)
                return document
            except PermissionError as exc:
                if attempt == 4:
                    print(f'[WARN] Cannot update detections.json (file in use): {exc}')
                    return None
                time.sleep(0.05)
    except OSError as exc:
        print(f'[WARN] JSON write failed: {exc}')
        return None
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def main():
    """Application entry point: choose offline image or live Manta camera.

    Loads the registry and detector once, then processes frames. In live
    mode the latest detection result is exported at ``--json-interval``
    seconds; ``S`` saves an additional timestamped image/JSON snapshot,
    and ``Q`` or Escape exits.

    Image coordinates are pixels, NOT calibrated robot millimeters."""
    p=argparse.ArgumentParser()
    p.add_argument('--registry',default=str(ROOT/'config/components.json'))
    p.add_argument('--camera-id',default=None,help='Optional; by default use auto-discovery with one camera')
    p.add_argument('--exposure-us',type=float,default=5000)
    p.add_argument('--scale',type=float,default=.85)
    p.add_argument('--image',help='Offline test image: skips VmbPy camera connection')
    p.add_argument('--json-interval',type=float,default=0.2,help='Minimum seconds between live JSON writes')
    args=p.parse_args()
    detector=ComponentDetector(args.registry)
    output_dir=ROOT/'output';output_dir.mkdir(exist_ok=True)
    print(f'[READY] {len(detector.components)} registered marker IDs | {detector.config["dictionary"]}')
    if args.json_interval <= 0:
        p.error('--json-interval must be positive')
    # Shared processing function for BOTH saved images and live camera frames.
    def process(frame):
        """Detect all registered components in one frame.

        Args:
            frame: A grayscale Mono8 (or compatible) numpy image.
        Returns:
            (found, unknown, processing_ms, overlay): recognized components,
            unregistered marker IDs, detector timing and annotated BGR frame.

        The processing time measures detection only, not drawing, streaming
        or JSON disk writes."""
        start=time.perf_counter()
        found,unknown=detector.detect(frame)
        ms=(time.perf_counter()-start)*1000
        overlay=draw_detections(frame,found,unknown)
        # Debug overlay: cyan boxes show ALL validly decoded ArUco markers,
        # even those not present in components.json (or rejected for pose).
        if detector.last_raw_ids:
            cv2.aruco.drawDetectedMarkers(
                overlay, detector.last_raw_corners,
                np.array(detector.last_raw_ids, dtype=np.int32).reshape(-1, 1),
                borderColor=(255, 255, 0)
            )
        cv2.putText(overlay, f'ArUco {ms:.1f} ms | Q quit | S save',
                    (12,58),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,240,20),2)
        return found,unknown,ms,overlay
    # Offline mode: validate detection without opening the Vimba camera SDK.
    if args.image:
        frame=cv2.imread(args.image,cv2.IMREAD_GRAYSCALE)
        if frame is None:raise FileNotFoundError(args.image)
        found,unknown,ms,overlay=process(frame)
        write_latest_detections(output_dir,found,unknown,ms)
        print(json.dumps({'detections':found},ensure_ascii=False,indent=2))
        target=output_dir/'offline_detection.png'
        cv2.imwrite(str(target),overlay)
        print('[SAVED]',target)
        return
    # Hardware imports happen only in live mode so offline tests need no VmbPy.
    from camera import MantaCamera
    from streaming import LatestFrameStream
    latest=LatestFrameStream()
    print('CONTROLS: S save image+JSON | Q/ESC quit | Close Viewer before running')
    last_print=0
    last_json_write=0.0
    last_json_reported_count=None
    last_json_warning=0.0
    print('[OUTPUT] Live results:', output_dir/'detections.json')
    with MantaCamera(camera_id=args.camera_id) as camera:
        camera.configure(exposure_us=args.exposure_us,gain=None)
        camera.cam.start_streaming(handler=latest.callback,buffer_count=5)
        try:
            while True:
                frame=latest.get(timeout=.5)
                if frame is None:
                    if (cv2.waitKey(1)&255) in (ord('q'),27):break
                    continue
                found,unknown,ms,overlay=process(frame)
                write_time=time.monotonic()
                # Update the SAME JSON file periodically; do not grow a history log.
                if write_time-last_json_write >= args.json_interval:
                    saved = write_latest_detections(output_dir, found, unknown, ms)
                    if saved is not None:
                        # Advance the timer only after a successful disk write.
                        last_json_write=time.monotonic()
                        count=(saved['detected_count'], len(saved['unknown_marker_ids']))
                        if count != last_json_reported_count:
                            print(f"[JSON OK] saved={saved['detected_count']} components, "
                                  f"unknown={len(saved['unknown_marker_ids'])}, "
                                  f"time={saved['timestamp_utc']}", flush=True)
                            last_json_reported_count=count
                    elif write_time-last_json_warning >= 3.0:
                        # The writer already printed the Windows exception.
                        print('[JSON WARN] Camera is still running, but detections.json was NOT updated.',
                              flush=True)
                        last_json_warning=write_time
                if args.scale!=1:
                    scaled=cv2.resize(overlay,None,fx=args.scale,fy=args.scale)
                else:scaled=overlay
                cv2.imshow('Manta ArUco - Multi Component Detection',scaled)
                now=time.monotonic()
                if found and now-last_print>2:
                    last_print=now
                    print('[DETECTED]', ' | '.join(f"{d['component_id']} {d['name']} ({d['center_px']}) angle={d['orientation_deg']}" for d in found))
                key=cv2.waitKey(1)&255
                if key in (ord('q'),27):break
                # Manual snapshot is separate from automatically refreshed detections.json.
                if key==ord('s'):
                    stamp=time.strftime('%Y%m%d_%H%M%S')
                    cv2.imwrite(str(output_dir/f'aruco_{stamp}.png'),overlay)
                    (output_dir/f'aruco_{stamp}.json').write_text(json.dumps({'detections':found},ensure_ascii=False,indent=2),encoding='utf-8')
                    print('[SAVED] image and JSON:',stamp)
        finally:
            camera.cam.stop_streaming()
            cv2.destroyAllWindows()

if __name__=='__main__':main()
