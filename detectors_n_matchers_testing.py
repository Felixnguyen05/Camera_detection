import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# Asynchronous grab
import time
from vmbpy import *

detector_option = "STARnBRIEF"

MIN_INLIERS = 12      # Define this next to MIN_MATCH_COUNT; must be well above 4
MIN_AREA_RATIO = 0.25 # Projected box area vs. original query box area
MAX_AREA_RATIO = 4.0

MIN_MATCH_COUNT = 10 # Minimum number of good matches before proceeding
Lowes_ratio = 0.85 # Test changing the ratio?
SPP = 2 # Second(s) per picture
Match_num = 5 # Max number of same type components in a picture

def query_init():
    __ = []
    
    #query images and their names
    query = [
        [cv.imread('Test_images/BBWhite.png',cv.IMREAD_GRAYSCALE), "BBWhite"],
        [cv.imread('Test_images/PlateWhite1.png',cv.IMREAD_GRAYSCALE), "PlateWhite"], 
        [cv.imread('Test_images/VoltWhite.png',cv.IMREAD_GRAYSCALE), "VoltWhite"], 
        [cv.imread('Test_images/ThinPlate1.png',cv.IMREAD_GRAYSCALE), "ThinPlate"],
        [cv.imread('Test_images/Current1.png',cv.IMREAD_GRAYSCALE), "Current"],
        [cv.imread('Test_images/Standoff1.png',cv.IMREAD_GRAYSCALE), "Standoff"]
    ]

    #init feature detector and find features of query images
    if detector_option == "SIFT":
        detector = cv.SIFT_create()
    elif detector_option == "SURF":
        detector = cv.xfeatures2d.SURF_create()
    elif detector_option == "STARnBRIEF":
        detector = cv.xfeatures2d.StarDetector_create()
        brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    qkp = []
    qdes = []
    for n in range(len(query)):
        qkp.append([])
        qdes.append([])

        if detector_option == ("SIFT" or "SURF"):
            qkp[n], qdes[n] = detector.detectAndCompute(query[n][0],None)
        
        elif detector_option == "STARnBRIEF":
            qkp[n] = detector.detect(query[n][0], None)
            qkp[n], qdes[n] = brief.compute(query[n][0], qkp[n])
            
    if detector_option == ("SIFT" or "SURF"):
        return qkp, qdes, query, detector, __
            
    elif detector_option == "STARnBRIEF":
        return qkp, qdes, query, detector, brief

def matcher(qdes, detector, brief, train, i, j):
    matches = []
    good = []
    bf = cv.BFMatcher()
    tkp = []
    tdes = []


    if detector_option == ("SIFT" or "SURF"):
        tkp, tdes = detector.detectAndCompute(train[j],None)
            
    elif detector_option == "STARnBRIEF":
        tkp = detector.detect(train[j], None)
        tkp, tdes = brief.compute(train[j], tkp)

    print(f"qdes = {qdes} tdes = {tdes}")
    # Change k=2 to k=5 so a single query feature can match multiple objects
    if qdes is not None and tdes is not None and len(tdes) >= Match_num:
        matches = bf.knnMatch(qdes[i], tdes, k=Match_num)
    else:
        matches = []

    # STARTOF AI assisted coding - Adding the ability to match multiple of
    # the same type of components. Unsuccesful, but solution does lead to 
    # increased accuracy in the resulting bounding boxes, thus kept.

    # Compare candidate matches against the last match in the group
    for n in matches:
        if len(n) == Match_num:
            # Check the top matches against the 5th match (match_group[-1])
            for m in n[:-1]:
                if m.distance < Lowes_ratio * n[-1].distance:
                    good.append([m]) #ENDOF AI assisted coding
    return good, tkp

def main():
    qkp, qdes, query, detector, brief = query_init()
    #print(f"qkp = {qkp},    qdes = {qdes}")

    train = [
        cv.imread('Test_images/Plates1.png',cv.IMREAD_GRAYSCALE), 
        cv.imread('Test_images/WhiteA6.png',cv.IMREAD_GRAYSCALE), 
        cv.imread('Test_images/WhiteA1.png',cv.IMREAD_GRAYSCALE),
        cv.imread('Test_images/Multi1.png',cv.IMREAD_GRAYSCALE)
    ]

    for i in range(len(query)):
        good, tkp = matcher(qdes, detector, brief, train, i, 0)
        print(f"good match = {good}")

    if len(good)>=MIN_MATCH_COUNT:
        print(f"enough good matches")

if __name__ == '__main__':
    main()