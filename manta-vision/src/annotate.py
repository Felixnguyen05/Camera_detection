"""Manual multi-instance polygon segmentation annotator, zero cloud service.

python src/annotate.py --split train
  Left-click: add polygon vertex | Right-click: undo vertex | Enter: commit
  1/2: select Speaker/Electronic Module class | S: save all labels
  N: next image | P: previous | U: remove last completed polygon | Q: quit

YOLO segmentation labels represent polygons, not topological holes. Label visible
outer component boundaries accurately; internal feature masks require another task.
"""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from vision.geometry import polygons_to_line

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {0:'Speaker',1:'Electronic Module'}


def read_existing(label_path, width, height):
    """Load existing YOLO normalized polygons to avoid accidental label loss."""
    existing=[]
    if not label_path.exists():return existing
    for row in label_path.read_text().splitlines():
        bits=row.split()
        if not bits:continue
        try:
            klass=int(bits[0]); vals=np.array([float(v) for v in bits[1:]], dtype=np.float64)
        except ValueError:continue
        if len(vals)<6 or len(vals)%2:continue
        xy=vals.reshape(-1,2)
        pts=np.column_stack((np.rint(xy[:,0]*width),np.rint(xy[:,1]*height))).astype(int)
        existing.append((klass,pts.tolist()))
    return existing


def annotate_one(path: Path, labels: Path, first_class=0):
    """Edit one image; returns next/prev/quit action or saves on S."""
    image=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if image is None:raise ValueError(f'Cannot open image: {path}')
    h,w=image.shape[:2]
    label_path=labels/f'{path.stem}.txt'
    completed=read_existing(label_path,w,h)
    current=[]
    selected=first_class
    window='Polygon labeling - see console controls'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    def on_mouse(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            current.append([int(x),int(y)])
        if event==cv2.EVENT_RBUTTONDOWN and current:
            current.pop()
    cv2.setMouseCallback(window,on_mouse)
    while True:
        view=image.copy()
        for idx,(cls,points) in enumerate(completed):
            poly=np.array(points,dtype=np.int32)
            cv2.polylines(view,[poly],True,(20,210,20),2)
            cv2.putText(view,f'{CLASSES.get(cls,cls)} #{idx+1}',tuple(poly[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,.6,(30,240,60),2)
        if current:
            cv2.polylines(view,[np.array(current,np.int32)],False,(0,220,255),2)
            for point in current:cv2.circle(view,tuple(point),3,(0,0,255),-1)
        cv2.putText(view,f'{path.name} | class {selected} {CLASSES[selected]} | polygons {len(completed)}',
                    (12,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
        cv2.putText(view,'Left add | Right undo | Enter commit | 1/2 class | S save | N next | P prev | U undo | Q',
                    (12,58),cv2.FONT_HERSHEY_SIMPLEX,.46,(0,255,255),1)
        cv2.imshow(window,view)
        key=cv2.waitKey(30)&255
        if key in (ord('1'),ord('2')):
            selected=int(chr(key))-1
        elif key in (13,10) and len(current)>=3:
            completed.append((selected,list(current)))
            current.clear()
        elif key==ord('u') and completed:
            completed.pop()
        elif key==ord('s'):
            labels.mkdir(parents=True,exist_ok=True)
            lines=[polygons_to_line(points,h,w,klass) for klass,points in completed]
            label_path.write_text('\n'.join(lines)+'\n' if lines else '',encoding='utf-8')
            print(f'[SAVED] {label_path.name} | {len(lines)} instances')
        elif key in (ord('n'),ord('p'),ord('q'),27):
            cv2.destroyWindow(window)
            return {ord('n'):1,ord('p'):-1,ord('q'):0,27:0}[key]


def main():
    """Iterate images in chosen split; explicit S saves each image's polygons."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['train','val'],default='train')
    args=p.parse_args()
    folder=ROOT/'dataset'/'images'/args.split
    labels=ROOT/'dataset'/'labels'/args.split
    images=sorted([x for x in folder.glob('*') if x.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'}])
    if not images:raise SystemExit(f'No images in {folder}. Run capture_dataset.py or copy images first.')
    i=0
    while 0<=i<len(images):
        step=annotate_one(images[i],labels)
        if step==0:break
        i=max(0,min(len(images)-1,i+step))

if __name__=='__main__':main()
