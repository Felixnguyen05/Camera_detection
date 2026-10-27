"""ONE-TIME component registration, from live camera (R capture+ROI), or saved image (--image).
Saved template persists across runs. The ROI is a rectangular crop, NOT precise segmentation.
"""
import argparse
import time
from pathlib import Path
import cv2
from camera import MantaCamera
from streaming import LatestFrameStream
from feature_matching import mono8

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT/'templates'/'component.png'
CAMERA_ID = None
EXPOSURE_US = 5000


def register(image):
    gray = mono8(image)
    print('Drag a rectangle around the ENTIRE component (including both mounting tabs). ENTER confirms; C cancels.')
    x,y,w,h = [int(v) for v in cv2.selectROI('Register component - select full part',gray,False,False)]
    cv2.destroyWindow('Register component - select full part')
    if w<30 or h<30:
        print('[CANCELLED] No valid ROI')
        return False
    TEMPLATE.parent.mkdir(parents=True,exist_ok=True)
    crop = gray[y:y+h,x:x+w]
    if not cv2.imwrite(str(TEMPLATE),crop):
        raise RuntimeError(f'Cannot write {TEMPLATE}')
    print(f'[REGISTERED] {TEMPLATE} | {w}x{h} | will be loaded automatically by main.py')
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',help='Use existing full-res image instead of opening the camera')
    args=parser.parse_args()
    if args.image:
        img=cv2.imread(args.image,cv2.IMREAD_GRAYSCALE)
        if img is None: raise FileNotFoundError(args.image)
        register(img)
        cv2.destroyAllWindows()
        return
    latest=LatestFrameStream()
    print('REGISTRATION | R: freeze frame and select component | S: save full frame | Q: quit')
    try:
        with MantaCamera(camera_id=CAMERA_ID) as camera:
            camera.configure(exposure_us=EXPOSURE_US,gain=None)
            camera.cam.start_streaming(handler=latest.callback,buffer_count=5)
            try:
                while True:
                    frame=latest.get(.5)
                    if frame is None:
                        if (cv2.waitKey(1)&0xff) in (ord('q'),27): break
                        continue
                    cv2.imshow('Manta - One-time registration',frame)
                    key=cv2.waitKey(1)&0xff
                    if key in (ord('q'),27):break
                    if key==ord('r'):
                        if register(frame):break
                    if key==ord('s'):
                        out=ROOT/'output';out.mkdir(exist_ok=True)
                        target=out/f'capture_{time.time_ns()}.png'
                        cv2.imwrite(str(target),frame)
                        print('[SAVED]',target)
            finally:
                camera.cam.stop_streaming()
    finally:
        cv2.destroyAllWindows()


if __name__=='__main__':main()
