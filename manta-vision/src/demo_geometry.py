"""Offline SYNTHETIC pipeline test without trained weights or physical camera.

python src/demo_geometry.py
This proves mask->contour->pose->JSON, NOT trained model performance.
"""
from pathlib import Path
import sys
import cv2
import numpy as np
from vision.geometry import mask_geometry
from vision.outputs import make_document, safe_write_json, save_masks
from vision.visualize import render
ROOT=Path(__file__).resolve().parents[1]

def main():
    """Create two independently segmented shapes and demonstrate export."""
    out=ROOT/'output'/'synthetic_demo';out.mkdir(parents=True,exist_ok=True)
    h,w=720,1040
    gray=np.full((h,w),180,dtype=np.uint8)
    masks=[];dets=[]
    # Speaker-like tabbed geometry, not a real object or recognition result.
    for idx,(cx,cy,deg,klass,name) in enumerate([(300,340,-25,0,'Speaker'),(735,370,35,1,'Electronic Module')]):
        mask=np.zeros((h,w),np.uint8)
        canvas=np.zeros((240,300),np.uint8)
        cv2.rectangle(canvas,(15,110),(285,130),255,-1)
        cv2.rectangle(canvas,(70,55),(230,185),255,-1)
        rotation=cv2.getRotationMatrix2D((150,120),deg,1.0)
        rot=cv2.warpAffine(canvas,rotation,(300,240),flags=cv2.INTER_NEAREST)
        y0,x0=cy-120,cx-150
        mask[y0:y0+240,x0:x0+300]=rot
        # Separate hole to verify contour hierarchy extraction.
        cv2.circle(mask,(cx,cy),18,0,-1)
        gray[mask>0]=70
        geometry=mask_geometry(mask)
        det={'instance_index':idx,'class_id':klass,'name':name,
             'confidence':1.0,'detection_method':'SYNTHETIC_TEST_NOT_MODEL',
             **geometry,'calibrated_robot_pose':False}
        masks.append(mask);dets.append(det)
    save_masks(out/'masks',masks,dets,'synthetic')
    doc=make_document(dets,0,source='synthetic_masks',model_path='none',frame_shape=gray.shape)
    safe_write_json(out/'detections.json',doc)
    cv2.imwrite(str(out/'preview.png'),render(gray,dets,masks,0))
    print('[SYNTHETIC PASS] Saved',out/'preview.png', 'and',out/'detections.json')
if __name__=='__main__':main()
