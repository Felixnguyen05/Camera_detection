import argparse
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--name', default='bracket')
    args = parser.parse_args()
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(args.image)
    print('Select object ROI with a small background margin, ENTER to confirm')
    x, y, w, h = [int(v) for v in cv2.selectROI('Select object', image, False, False)]
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        print('Selection cancelled')
        return
    Path('templates').mkdir(exist_ok=True)
    output = Path('templates') / (args.name + '.png')
    if not cv2.imwrite(str(output), image[y:y+h, x:x+w]):
        raise RuntimeError('Cannot save cropped template')
    print('Template saved:', output)


if __name__ == '__main__':
    main()
