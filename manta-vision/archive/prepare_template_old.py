"""Crop a REFERENCE image tightly around the complete component."""
import argparse
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Saved full-resolution Mono8 camera frame')
    args = parser.parse_args()
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(args.image)
    print('Select the WHOLE component with little surrounding background. ENTER confirms; C cancels.')
    x, y, w, h = cv2.selectROI('Select full component', img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        print('Canceled')
        return
    dest = ROOT / 'templates' / 'component.png'
    dest.parent.mkdir(exist_ok=True)
    if not cv2.imwrite(str(dest), img[y:y+h, x:x+w]):
        raise RuntimeError(f'Could not write {dest}')
    print('[SAVED]', dest)


if __name__ == '__main__':
    main()
