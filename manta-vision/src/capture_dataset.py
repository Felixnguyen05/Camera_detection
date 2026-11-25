"""Capture ORIGINAL Mono8 images for training; never draw overlays onto data.

python src/capture_dataset.py --split train --session day1 --exposure-us 5000
S: save one frame; A: toggle auto sample every --interval seconds; Q: exit.
"""
from pathlib import Path
import argparse
import time
import cv2
from camera import MantaCamera
from streaming import LatestFrameStream
ROOT = Path(__file__).resolve().parents[1]


def main():
    """Save raw full-resolution camera frames in a selected dataset split."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['train','val'], default='train')
    p.add_argument('--session', default='session01', help='Name acquisition session to avoid accidental leakage')
    p.add_argument('--camera-id', default=None)
    p.add_argument('--exposure-us', type=float, default=5000)
    p.add_argument('--interval', type=float, default=1.5)
    args = p.parse_args()
    if args.interval <= 0:
        p.error('--interval must be > 0')
    folder = ROOT/'dataset'/'images'/args.split
    folder.mkdir(parents=True,exist_ok=True)
    stream = LatestFrameStream()
    serial = 0
    auto = False
    last = 0.0
    print(f'[SAVE TO] {folder} | S manual | A auto interval | Q quit')
    with MantaCamera(camera_id=args.camera_id) as cam:
        cam.configure(exposure_us=args.exposure_us)
        cam.cam.start_streaming(stream.callback, buffer_count=5)
        try:
            while True:
                image = stream.get(0.4)
                if image is None:
                    if cv2.waitKey(1)&255 in (ord('q'),27):
                        break
                    continue
                view = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
                cv2.putText(view,f'S save | A auto={auto} | Q quit', (10,28),
                            cv2.FONT_HERSHEY_SIMPLEX,.75,(40,255,40),2)
                cv2.imshow('Dataset acquisition - raw Mono8', view)
                key = cv2.waitKey(1)&255
                if key in (ord('q'),27):
                    break
                if key == ord('a'):
                    auto = not auto
                now = time.monotonic()
                if key == ord('s') or (auto and now-last>=args.interval):
                    stamp = time.strftime('%Y%m%d_%H%M%S')
                    serial += 1
                    path = folder/f'{args.session}_{stamp}_{serial:04d}.png'
                    if cv2.imwrite(str(path), image):
                        print('[SAVED]', path.name)
                    last = now
        finally:
            cam.cam.stop_streaming()
            cv2.destroyAllWindows()

if __name__ == '__main__':main()
