"""Evaluate real component mask boundaries on held-out YOLO-polygon labels.

python src/evaluate.py --weights models/best.pt --imgsz 960
Reported metrics are test-set estimates; never infer actual robot pose accuracy.
"""
import argparse
from pathlib import Path
import json
from time import perf_counter
import cv2
import numpy as np
from vision.inference import ComponentSegmenter
ROOT=Path(__file__).resolve().parents[1]


def load_gt(label:Path,height:int,width:int):
    """Rasterize labelled polygons to per-instance masks for fair comparisons."""
    items=[]
    if not label.exists():raise FileNotFoundError(label)
    for row in label.read_text().splitlines():
        bits=row.split()
        if not bits:continue
        klass=int(bits[0]);pts=np.asarray([float(v) for v in bits[1:]],dtype=float).reshape(-1,2)
        pts[:,0]*=width;pts[:,1]*=height
        mask=np.zeros((height,width),dtype=np.uint8)
        cv2.fillPoly(mask,[np.rint(pts).astype(np.int32)],1)
        items.append((klass,mask))
    return items


def iou(a,b):
    """Binary intersection over union, zero when either/both masks empty."""
    x=a>0;y=b>0
    u=int(np.count_nonzero(x|y))
    return float(np.count_nonzero(x&y))/u if u else 0.0


def boundary_f1(a,b,tolerance=3):
    """Symmetric contour F1 with pixel tolerance (not exact-pixel boundary match)."""
    a=np.uint8(a>0);b=np.uint8(b>0)
    ker=np.ones((3,3),np.uint8)
    edge_a=a-cv2.erode(a,ker,iterations=1)
    edge_b=b-cv2.erode(b,ker,iterations=1)
    na=int(np.count_nonzero(edge_a));nb=int(np.count_nonzero(edge_b))
    if not na or not nb:return 0.0
    window=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*tolerance+1,2*tolerance+1))
    tolerant_b=cv2.dilate(edge_b,window)
    tolerant_a=cv2.dilate(edge_a,window)
    precision=np.count_nonzero(edge_a&tolerant_b)/na
    recall=np.count_nonzero(edge_b&tolerant_a)/nb
    return float(2*precision*recall/(precision+recall)) if precision+recall else 0.0


def evaluate(segmenter,split='val',threshold=.50,limit=0):
    """Greedy class-aware one-to-one instance matching; report mask metrics."""
    images=sorted(p for p in (ROOT/'dataset'/'images'/split).glob('*')
                  if p.suffix.lower() in ('.png','.jpg','.jpeg'))
    if limit>0:images=images[:limit]
    totals={'images':0,'ground_truth_instances':0,'predicted_instances':0,
            'true_positives':0,'false_positives':0,'false_negatives':0}
    matched_ious=[];matched_boundaries=[];latencies=[]
    for image in images:
        frame=cv2.imread(str(image),cv2.IMREAD_GRAYSCALE)
        if frame is None:continue
        h,w=frame.shape[:2]
        gt=load_gt(ROOT/'dataset'/'labels'/split/(image.stem+'.txt'),h,w)
        dets,masks,ms=segmenter.predict(frame)
        latencies.append(ms)
        totals['images']+=1
        totals['ground_truth_instances']+=len(gt)
        totals['predicted_instances']+=len(dets)
        candidates=[]
        for pi,(det,pred) in enumerate(zip(dets,masks)):
            for gi,(klass,target) in enumerate(gt):
                if int(det['class_id'])==int(klass):
                    score=iou(pred,target)
                    if score>=threshold:
                        candidates.append((score,pi,gi))
        candidates.sort(reverse=True)
        paired_pred=set();paired_gt=set()
        for score,pi,gi in candidates:
            if pi in paired_pred or gi in paired_gt:continue
            paired_pred.add(pi);paired_gt.add(gi)
            matched_ious.append(score)
            matched_boundaries.append(boundary_f1(masks[pi],gt[gi][1]))
        totals['true_positives']+=len(paired_pred)
        totals['false_positives']+=len(dets)-len(paired_pred)
        totals['false_negatives']+=len(gt)-len(paired_gt)
        print(f'[VAL] {image.name}: gt={len(gt)} pred={len(dets)} matched={len(paired_pred)}')
    tp=totals['true_positives'];fp=totals['false_positives'];fn=totals['false_negatives']
    totals.update({
        'matching_iou_threshold':threshold,
        'instance_precision':round(tp/max(1,tp+fp),4),
        'instance_recall':round(tp/max(1,tp+fn),4),
        'mean_matched_mask_iou':round(float(np.mean(matched_ious)),4) if matched_ious else None,
        'mean_matched_boundary_f1_tol3px':round(float(np.mean(matched_boundaries)),4) if matched_boundaries else None,
        'mean_processing_ms':round(float(np.mean(latencies)),2) if latencies else None,
        'note':'IoU/BoundaryF1 averaged over matched detections only; unmatched counted in precision/recall. No position/orientation ground truth evaluated.'
    })
    return totals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,default=ROOT/'models/best.pt')
    p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--confidence',type=float,default=.35)
    p.add_argument('--device',default='auto')
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    model=ComponentSegmenter(args.weights,confidence=args.confidence,imgsz=args.imgsz,device=args.device)
    results=evaluate(model,limit=args.limit)
    target=ROOT/'output'/'validation_metrics.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('[METRICS]',json.dumps(results,indent=2))
    print('[SAVED]',target)
if __name__=='__main__':main()
