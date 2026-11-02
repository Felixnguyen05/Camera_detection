"""Create print-ready high-resolution ArUco PNGs. Print at 100% scale to requested millimeters."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from aruco_detection import read_registry

ROOT = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--registry',default=str(ROOT/'config/components.json'))
    p.add_argument('--dpi', type=int, default=300)
    args=p.parse_args()
    cfg, comps=read_registry(args.registry)
    dictionary=cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,cfg['dictionary']))
    size_mm=float(cfg['marker_size_mm'])
    side_px=round(size_mm/25.4*args.dpi)
    side_px=max(side_px, 140)
    out=ROOT/'markers';out.mkdir(exist_ok=True)
    quiet=max(24,side_px//5)
    # The BLACK square occupies marker_size_mm; white margin is OUTSIDE it.
    for marker_id,item in sorted(comps.items()):
        img=cv2.aruco.generateImageMarker(dictionary, marker_id,side_px)
        sheet=cv2.copyMakeBorder(img, quiet,quiet,quiet,quiet,cv2.BORDER_CONSTANT,value=255)
        path=out/f"marker_{marker_id:02d}_{item['component_id']}.png"
        cv2.imwrite(str(path),sheet)
        print(f'[SAVED] {path.name} | black marker size at {args.dpi} DPI: {size_mm:.1f} mm (print at 100%)')
    print('IMPORTANT: PNGs do not guarantee printer scale. Measure printed BLACK square with a ruler.')

if __name__=='__main__':main()
