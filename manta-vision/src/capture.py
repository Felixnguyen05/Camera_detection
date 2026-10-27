"""Capture a sharp full-resolution Mono8 reference image from Manta G-125B.
Usage: python src/capture.py   (S save, Q quit)
"""
import time
from pathlib import Path
import cv2
from camera import MantaCamera
from streaming import LatestFrameStream

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
CAMERA_ID = 'DEV_000F314C8213'
EXPOSURE_US = 5000


def main():
    latest = LatestFrameStream()
    print('CAPTURE | S: save full-resolution frame | Q / ESC: quit')
    try:
        with MantaCamera(camera_id=CAMERA_ID) as camera:
            camera.configure(exposure_us=EXPOSURE_US, gain=None)
            camera.cam.start_streaming(handler=latest.callback, buffer_count=5)
            try:
                while True:
                    frame = latest.get(0.5)
                    if frame is None:
                        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                            break
                        continue
                    cv2.imshow('Manta - Capture reference', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        break
                    if key == ord('s'):
                        dest = OUT / f'capture_{time.time_ns()}.png'
                        if not cv2.imwrite(str(dest), frame):
                            raise RuntimeError(f'Unable to save {dest}')
                        print('[SAVED]', dest)
            finally:
                camera.cam.stop_streaming()
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
