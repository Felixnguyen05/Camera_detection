"""Offline test: python src/test_feature_matching.py --image output/example.png [--algorithm SIFT]"""
import argparse
from pathlib import Path
import cv2
from feature_matching import FeatureMatcher
from visualization import draw

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--algorithm',default='SIFT',choices=['SIFT','ORB']);args=p.parse_args()
    img=cv2.imread(args.image,cv2.IMREAD_GRAYSCALE)
    if img is None:raise FileNotFoundError(args.image)
    fm=FeatureMatcher(ROOT/'templates'/'component.png',algorithm=args.algorithm)
    det=fm.detect(img)
    print('[RESULT]',det)
    output=draw(img,det,args.algorithm,0,0)
    output_path=ROOT/'output'/'offline_result.png';output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path),output)
    print('[SAVED]',output_path)
    cv2.imshow('Offline feature matching',output);cv2.waitKey(0);cv2.destroyAllWindows()
if __name__=='__main__':main()
