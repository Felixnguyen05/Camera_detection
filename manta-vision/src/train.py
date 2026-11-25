"""Fine-tune segmentation weights for real industrial component masks.

python src/train.py --epochs 80 --imgsz 960 --device auto
The first run downloads the *generic* pretrained model, NOT an industrial model.
"""
from pathlib import Path
import argparse
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1]

def main():
    """Audit the annotated dataset, train, validate, and copy best.pt into models/."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='yolo26n-seg.pt')
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--device',default='auto',help='auto, cpu, or CUDA index 0')
    p.add_argument('--workers',type=int,default=0,help='Windows-safe default')
    p.add_argument('--patience',type=int,default=25)
    p.add_argument('--run-name',default='components_seg')
    args=p.parse_args()
    if args.epochs<1 or args.batch<1:p.error('epochs and batch must be positive')
    from audit_dataset import audit
    if not audit(ROOT/'dataset'):
        raise SystemExit('Fix dataset problems before training')
    from ultralytics import YOLO
    import torch
    if args.device=='auto':
        device=0 if torch.cuda.is_available() else 'cpu'
    else:
        device=int(args.device) if args.device.isdigit() else args.device
    yaml_path=ROOT/'dataset'/'data.yaml'
    # Absolute path independent of current working directory, Windows friendly.
    data={'path':(ROOT/'dataset').resolve().as_posix(),
          'train':'images/train', 'val':'images/val',
          'names':{0:'Speaker',1:'Electronic Module'}}
    yaml_path.write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    print('[TRAIN] device:',device,'| model:',args.model,'| data:',yaml_path)
    model=YOLO(args.model)
    if model.task!='segment':raise ValueError('Choose a -seg model, not object detection weights')
    results=model.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz,
                        batch=args.batch, device=device, workers=args.workers,
                        patience=args.patience, project=str(ROOT/'runs'),
                        name=args.run_name, exist_ok=False, verbose=True,
                        degrees=180.0, fliplr=.5, flipud=.5,
                        hsv_h=.0, hsv_s=.0, hsv_v=.0)
    best=Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError('Training did not produce best.pt')
    target=ROOT/'models'/'best.pt'
    import shutil
    shutil.copy2(best,target)
    print('[TRAIN DONE] weights copied to:',target)
    # Validation on val split; report detection and segmentation mAP honestly.
    metrics=YOLO(str(target)).val(data=str(yaml_path),imgsz=args.imgsz,device=device,
                                   project=str(ROOT/'runs'),name='validation')
    print('[VALIDATION] segmentation mAP50:',float(metrics.seg.map50),
          '| segmentation mAP50-95:',float(metrics.seg.map))
if __name__=='__main__':main()
