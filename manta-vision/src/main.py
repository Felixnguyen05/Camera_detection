"""Production-style demonstration: one command for continuous known-component detection.
No thresholding, background setup or manual annotations at runtime.
"""
import time
from pathlib import Path
import cv2
from camera import MantaCamera
from streaming import LatestFrameStream
from feature_matching import FeatureMatcher
from visualization import draw

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT/'templates'/'component.png'
OUT = ROOT/'output'
CAMERA_ID = 'DEV_000F314C8213'
EXPOSURE_US = 5000
DISPLAY_SCALE = .75


def main():
    if not TEMPLATE.is_file():
        raise FileNotFoundError('No registered component. First run: python src/register_component.py')
    matchers={}
    for method in ('SIFT','ORB'):
        try:
            matchers[method]=FeatureMatcher(TEMPLATE,algorithm=method)
            print(f'[READY] {method}: {len(matchers[method].ref_kp)} reference keypoints')
        except (ValueError,AttributeError) as e:
            print('[UNAVAILABLE]',method,e)
    if not matchers:
        raise RuntimeError('Registration has insufficient keypoints for SIFT and ORB. Use a sharper whole-object image.')
    method='SIFT' if 'SIFT' in matchers else 'ORB'
    latest=LatestFrameStream()
    OUT.mkdir(exist_ok=True)
    previous_time=time.perf_counter(); previous_count=0; rx_fps=0.0
    print('AUTO DETECTION | 1: SIFT | 2: ORB | S: save frame | Q/ESC: exit')
    try:
        with MantaCamera(camera_id=CAMERA_ID) as camera:
            camera.configure(exposure_us=EXPOSURE_US,gain=None)
            camera.cam.start_streaming(handler=latest.callback,buffer_count=5)
            try:
                while True:
                    image=latest.get(timeout=.5)
                    if image is None:
                        if cv2.waitKey(1)&0xff in (ord('q'),27):break
                        continue
                    started=time.perf_counter()
                    result=matchers[method].detect(image)
                    elapsed_ms=(time.perf_counter()-started)*1000
                    now=time.perf_counter()
                    if now-previous_time>=1:
                        rx_fps=(latest.received-previous_count)/(now-previous_time)
                        previous_time=now;previous_count=latest.received
                    display=draw(image,result,method,elapsed_ms,rx_fps)
                    if DISPLAY_SCALE!=1:
                        display=cv2.resize(display,None,fx=DISPLAY_SCALE,fy=DISPLAY_SCALE,interpolation=cv2.INTER_AREA)
                    cv2.imshow('Manta - Automatic Component Detection',display)
                    key=cv2.waitKey(1)&0xff
                    if key in (ord('q'),27):break
                    elif key==ord('1') and 'SIFT' in matchers:
                        method='SIFT';print('[MODE] SIFT')
                    elif key==ord('2') and 'ORB' in matchers:
                        method='ORB';print('[MODE] ORB')
                    elif key==ord('s'):
                        target=OUT/f'detection_{time.time_ns()}.png'
                        if cv2.imwrite(str(target),image):print('[SAVED]',target)
            finally:
                camera.cam.stop_streaming()
    finally:
        cv2.destroyAllWindows()

if __name__=='__main__':main()
