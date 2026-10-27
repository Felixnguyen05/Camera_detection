"""Non-camera regression tests. Run: python tests/test_affine.py"""
from pathlib import Path
import sys, tempfile
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from feature_matching import FeatureMatcher, mono8


def test_mono():
    a=np.zeros((40,50,1),dtype=np.uint8)
    assert mono8(a).shape==(40,50)


def test_synthetic():
    rng=np.random.default_rng(42)
    ref=np.zeros((260,360),dtype=np.uint8)+125
    # Rich, asymmetric component-like geometry + text, helps validate estimated rotation.
    cv2.rectangle(ref,(22,65),(337,201),220,-1)
    cv2.rectangle(ref,(93,34),(269,226),180,-1)
    cv2.circle(ref,(181,130),78,45,12)
    cv2.circle(ref,(181,130),31,15,-1)
    for _ in range(140):
        x=int(rng.integers(24,330));y=int(rng.integers(40,218))
        cv2.circle(ref,(x,y),int(rng.integers(1,4)),int(rng.integers(15,250)),-1)
    cv2.putText(ref,'PART-42',(90,40),cv2.FONT_HERSHEY_SIMPLEX,.65,10,2)
    center=(ref.shape[1]/2,ref.shape[0]/2)
    mat=cv2.getRotationMatrix2D(center,28,1.1)
    mat[:,2]+=np.array([210,175])
    scene=cv2.warpAffine(ref,mat,(850,640),borderValue=125)
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'reference.png';cv2.imwrite(str(path),ref)
        for algorithm in ('SIFT','ORB'):
            matcher=FeatureMatcher(path,algorithm=algorithm)
            detection=matcher.detect(scene)
            print(algorithm, detection)
            assert detection is not None
            expected=cv2.transform(np.float32([[[center[0],center[1]]]]),mat)[0,0]
            assert np.linalg.norm(np.asarray(detection.center)-expected)<6
            assert abs(detection.scale-1.1)<.08
            # cv2 rotation angle sign is negative in downward-Y image coordinates
            angle_error=((detection.angle_deg+28+180)%360)-180
            assert abs(angle_error)<5

if __name__=='__main__':
    test_mono();test_synthetic();print('ALL TESTS PASSED')
