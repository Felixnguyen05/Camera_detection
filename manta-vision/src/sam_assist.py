"""OPTIONAL SAM2 mask-assisted single-instance annotation via manual rectangle.

python src/sam_assist.py --image dataset/images/train/photo.png --class-id 0
Select a tight rectangle around ONE object, SAM proposes a polygon, and you MUST
visually inspect the preview before pressing Y. Repeat with another object box.
SAM2 weights are downloaded on first run; no SAM inference runs in main.py.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
from vision.geometry import polygons_to_line
from annotate import ROOT


def main():
    """Prompt SAM 2 with ROI and append a reviewed polygon label."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--class-id',type=int,choices=[0,1],required=True)
    p.add_argument('--model',default='sam2.1_t.pt')
    p.add_argument('--device',default='auto')
    args=p.parse_args()
    from ultralytics import SAM
    image=cv2.imread(str(args.image))
    if image is None:raise FileNotFoundError(args.image)
    h,w=image.shape[:2]
    roi=cv2.selectROI('Select one component (Enter confirm)',image,fromCenter=False,showCrosshair=True)
    cv2.destroyWindow('Select one component (Enter confirm)')
    x,y,rw,rh=map(int,roi)
    if rw<5 or rh<5:raise SystemExit('No ROI selected')
    model=SAM(args.model)
    kwargs=dict(source=image,bboxes=[x,y,x+rw,y+rh],verbose=False)
    if args.device!='auto':kwargs['device']=args.device
    result=model.predict(**kwargs)[0]
    if result.masks is None:raise SystemExit('SAM returned no mask; try tighter box')
    full=result.masks.data[0].cpu().numpy()
    if full.shape[:2]!=(h,w):
        full=cv2.resize(full.astype(np.float32),(w,h),interpolation=cv2.INTER_NEAREST)
    mask=np.uint8(full>0.5)*255
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:raise SystemExit('No contour')
    contour=max(contours,key=cv2.contourArea)[:,0,:]
    if len(contour)<3:raise SystemExit('Invalid predicted contour')
    review=image.copy()
    cv2.polylines(review,[contour],True,(0,255,0),2)
    cv2.putText(review,'Y accept (after inspection) / N reject',(12,35),
                cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
    cv2.imshow('REVIEW SAM mask',review)
    key=cv2.waitKey(0)&255
    cv2.destroyAllWindows()
    if key!=ord('y'):
        print('[REJECTED] No annotation saved')
        return
    split=args.image.parent.name
    if split not in {'train','val'}:raise ValueError('Image must be in dataset/images/train or val')
    label_path=ROOT/'dataset'/'labels'/split/(args.image.stem+'.txt')
    label_path.parent.mkdir(parents=True,exist_ok=True)
    new_line=polygons_to_line(contour,h,w,args.class_id)
    with label_path.open('a',encoding='utf-8') as fp:fp.write(new_line+'\n')
    print('[SAVED] Appended SAM-assisted instance polygon:',label_path)

if __name__=='__main__':main()
