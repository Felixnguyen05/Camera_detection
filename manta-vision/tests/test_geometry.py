import sys
from pathlib import Path
import unittest
import tempfile
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vision.geometry import mask_geometry, normalize_gray, polygons_to_line, preprocess
from vision.outputs import make_document,safe_write_json

class TestGeometry(unittest.TestCase):
    def test_gray_normalization(self):
        img=np.zeros((50,40,1),dtype=np.uint8)
        self.assertEqual(normalize_gray(img).shape,(50,40))
        self.assertEqual(preprocess(img,'clahe').shape,(50,40))
        with self.assertRaises(ValueError): normalize_gray(np.zeros((12,12),dtype=np.float32))
    def test_mask_center_and_holes(self):
        mask=np.zeros((200,300),dtype=np.uint8)
        cv2.rectangle(mask,(55,80),(245,120),1,-1)
        cv2.circle(mask,(150,100),12,0,-1)
        geo=mask_geometry(mask)
        self.assertAlmostEqual(geo['center_px'][0],150,delta=0.2)
        self.assertAlmostEqual(geo['center_px'][1],100,delta=0.2)
        self.assertTrue(geo['axis_reliable'])
        self.assertTrue(geo['direction_ambiguous'])
        self.assertLess(min(geo['orientation_deg'],180-geo['orientation_deg']),2)
        self.assertGreaterEqual(len(geo['holes_contours_px']),1)
    def test_rotated_orientation(self):
        m=np.zeros((320,320),dtype=np.uint8)
        box=cv2.boxPoints(((160,160),(160,30),35)).astype(np.int32)
        cv2.fillPoly(m,[box],1)
        geom=mask_geometry(m)
        self.assertIsNotNone(geom)
        self.assertAlmostEqual(geom['orientation_deg'],35,delta=2)
    def test_symmetric_orientation_unreliable(self):
        mask=np.zeros((100,100),np.uint8)
        cv2.circle(mask,(50,50),30,1,-1)
        geom=mask_geometry(mask)
        self.assertFalse(geom['axis_reliable'])
        self.assertIsNone(geom['orientation_deg'])
    def test_multiple_instances_separated(self):
        a=np.zeros((150,350),np.uint8);b=a.copy()
        cv2.rectangle(a,(10,20),(100,60),1,-1)
        cv2.rectangle(b,(210,70),(320,110),1,-1)
        geos=[mask_geometry(x) for x in (a,b)]
        self.assertNotEqual(geos[0]['center_px'][0],geos[1]['center_px'][0])
    def test_polygon_and_json(self):
        p=[[0,0],[100,0],[100,50],[0,50]]
        row=polygons_to_line(p,200,300,1)
        self.assertTrue(row.startswith('1 '))
        doc=make_document([{'name':'test','center_px':[20,30]}],4.5,
                           source='unit_test',model_path='none',frame_shape=(150,300))
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'detections.json'
            self.assertTrue(safe_write_json(path,doc))
            self.assertIn('detected_count',path.read_text())
            self.assertFalse(doc['robot_pose_calibrated'])
    def test_empty_mask(self):
        self.assertIsNone(mask_geometry(np.zeros((100,100),np.uint8)))

if __name__=='__main__':unittest.main()
