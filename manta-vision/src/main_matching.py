import queue
import time
from pathlib import Path

import cv2
import numpy as np
from vmbpy import FrameStatus, PixelFormat

from camera import MantaCamera
from matching import ObjectMatcher

CAMERA_ID = 'DEV_000F314C8213'
TEMPLATE = Path('templates/bracket.png')
EXPOSURE_US = 5000
DARK_OBJECT = True  # change to False if object is brighter than background
MIN_AREA = 400
DISPLAY_SCALE = 0.8


def annotate(gray, detections, mode, ms, frames_received):
    view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(view, f'{mode.upper()} | {ms:.1f} ms | Received: {frames_received}',
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for d in detections:
        cx, cy = d['center']
        x, y, w, h = d['bbox']
        cv2.drawContours(view, [d['contour']], -1, (0, 220, 0), 2)
        cv2.rectangle(view, (x, y), (x+w, y+h), (255, 190, 0), 1)
        cv2.circle(view, (round(cx), round(cy)), 5, (0, 0, 255), -1)
        # Angle comes from rotating the reference mask, not the robot axes.
        theta = np.radians(d['angle_deg'])
        tip = (round(cx + 55 * np.cos(theta)), round(cy - 55 * np.sin(theta)))
        cv2.arrowedLine(view, (round(cx), round(cy)), tip, (255, 0, 255), 2)
        label = (f'({cx:.0f},{cy:.0f}) theta={d["angle_deg"]:.0f} '
                 f'Hu={d["shape_distance"]:.3f} Corr={d["template_score"]:.2f}')
        cv2.putText(view, label, (max(5, x), max(45, y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, (20, 240, 20), 1)
    return view


def main():
    if not TEMPLATE.is_file():
        raise FileNotFoundError(f'Missing {TEMPLATE}. Run prepare_template.py first.')
    matcher = ObjectMatcher(TEMPLATE, dark_object=DARK_OBJECT, min_area=MIN_AREA)
    latest = queue.Queue(maxsize=1)
    count = [0]

    def handler(cam, stream, frame):
        try:
            if frame.get_status() != FrameStatus.Complete:
                return
            if frame.get_pixel_format() != PixelFormat.Mono8:
                return
            image = frame.as_opencv_image().copy()
            count[0] += 1
            if latest.full():
                try:
                    latest.get_nowait()
                except queue.Empty:
                    pass
            try:
                latest.put_nowait(image)
            except queue.Full:
                pass
        finally:
            cam.queue_frame(frame)

    mode = 'shape'
    print('Controls: 1 shape | 2 template | S save | Q/ESC quit')
    with MantaCamera(camera_id=CAMERA_ID) as camera:
        camera.configure(exposure_us=EXPOSURE_US, gain=None)
        camera.cam.start_streaming(handler=handler, buffer_count=5)
        cv2.namedWindow('Manta detection', cv2.WINDOW_NORMAL)
        try:
            while True:
                try:
                    frame = latest.get(timeout=0.5)
                except queue.Empty:
                    print('[WARN] No complete frame received')
                    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                        break
                    continue
                started = time.perf_counter()
                detections, mask = matcher.detect(frame, method=mode)
                elapsed_ms = (time.perf_counter() - started) * 1000
                view = annotate(frame, detections, mode, elapsed_ms, count[0])
                if DISPLAY_SCALE != 1.0:
                    view = cv2.resize(view, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
                cv2.imshow('Manta detection', view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('1'):
                    mode = 'shape'
                elif key == ord('2'):
                    mode = 'template'
                elif key == ord('s'):
                    Path('output').mkdir(exist_ok=True)
                    file = Path('output') / f'detection_{time.time_ns()}.png'
                    cv2.imwrite(str(file), frame)
                    print('Saved', file)
                elif key in (ord('q'), 27):
                    break
        finally:
            camera.cam.stop_streaming()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
