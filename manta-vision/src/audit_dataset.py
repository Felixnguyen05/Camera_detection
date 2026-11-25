"""Check YOLO segmentation labels, counts and session leakage before training."""
from pathlib import Path
from collections import Counter
import argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def audit(root:Path):
    """Validate polygon syntax, all image-label pairs, and split/session overlap."""
    errors=[]; stats={}; sessions={}
    for split in ('train','val'):
        images=sorted(p for p in (root/'images'/split).glob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp'))
        labels=root/'labels'/split
        count=Counter(); session_set=set()
        for image in images:
            # Session is filename prefix before first '_YYYY...'; use distinct session names.
            session_set.add(image.stem.split('_')[0])
            label=labels/(image.stem+'.txt')
            if not label.exists():
                errors.append(f'[{split}] Missing label (empty file is allowed): {image.name}')
                continue
            for line_no,line in enumerate(label.read_text(encoding='utf-8').splitlines(),1):
                bits=line.split()
                if not bits:continue
                try:
                    cls=int(bits[0]); vals=np.array([float(x) for x in bits[1:]])
                    if cls not in (0,1):raise ValueError('invalid class (expected 0 or 1)')
                    if len(vals)<6 or len(vals)%2 or not np.isfinite(vals).all() or (vals<0).any() or (vals>1).any():
                        raise ValueError('invalid polygon coordinates')
                    count[cls]+=1
                except ValueError as exc:
                    errors.append(f'[{split}] {label.name}:{line_no}: {exc}')
        stats[split]=(len(images),count)
        sessions[split]=session_set
    overlap=sessions['train'] & sessions['val']
    if overlap:
        errors.append(f'Session name overlap between train and val: {sorted(overlap)}; split by acquisition session')
    for split,(n,counts) in stats.items():
        print(f'[{split.upper()}] images={n} speaker={counts[0]} module={counts[1]}')
        if n==0:errors.append(f'[{split}] no images')
    for error in errors:print('[ERROR]',error)
    print(f'[AUDIT] {len(errors)} issue(s)')
    return len(errors)==0

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=ROOT/'dataset');args=p.parse_args()
    raise SystemExit(0 if audit(args.dataset) else 1)
if __name__=='__main__':main()
