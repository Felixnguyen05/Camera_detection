import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aruco_detection import ComponentDetector, component_from_corners, to_mono8


class MarkerTests(unittest.TestCase):
    def setUp(self):
        self.detector=ComponentDetector(ROOT/'config/components.json')
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.gray=np.full((700,1000),215,dtype=np.uint8)
        marker_size=160
        for mid,x,y in ((0,90,120),(1,420,170),(2,740,380)):
            mark=cv2.aruco.generateImageMarker(dictionary,mid,marker_size)
            self.gray[y:y+marker_size,x:x+marker_size]=mark

    def test_mono8_singleton_channel(self):
        self.assertEqual(to_mono8(self.gray[...,None]).shape,self.gray.shape)

    def test_multiple_components_and_ids(self):
        detections,unknown=self.detector.detect(self.gray)
        self.assertEqual(unknown,[])
        self.assertEqual({d['marker_id'] for d in detections},{0,1,2})
        self.assertEqual({d['component_id'] for d in detections},{'SPK-001','SPK-002','MOD-001'})
        for d in detections:
            self.assertLess(abs(d['orientation_deg']),2.5)

    def test_offset_marker_to_component(self):
        item=dict(self.detector.components[0])
        item['marker_to_component_mm']=[10,0]
        corners=np.array([[100,100],[200,100],[200,200],[100,200]],np.float32)
        detected=component_from_corners(0,corners,item,20)
        self.assertAlmostEqual(detected['center_px'][0],200,delta=.2)
        self.assertAlmostEqual(detected['center_px'][1],150,delta=.2)

    def test_rotation(self):
        item=dict(self.detector.components[0])
        corners=np.array([[200,100],[200,200],[100,200],[100,100]],np.float32)
        detected=component_from_corners(0,corners,item,20)
        self.assertAlmostEqual(detected['orientation_deg'],90,delta=.3)

    def test_no_marker(self):
        found,unknown=self.detector.detect(np.full((500,500),210,np.uint8))
        self.assertEqual(found,[])
        self.assertEqual(unknown,[])

if __name__=='__main__':unittest.main()
