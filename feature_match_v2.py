import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# Asynchronous grab
import time
from vmbpy import *

with VmbSystem.get_instance() as vmb:
    cams = vmb.get_all_cameras()
    with cams[0] as cam:

        frame = cam.get_frame()
        frame_mono8 = frame.convert_pixel_format(PixelFormat.Mono8)
        frameCV = frame_mono8.as_opencv_image()
        start = time.time()
        while True: 
            end = time.time()
            if end - start > 2:
                frame = cam.get_frame()
                frame_mono8 = frame.convert_pixel_format(PixelFormat.Mono8)
                frameCV = frame_mono8.as_opencv_image()
                cv.destroyAllWindows()
                start = time.time()

            #query images
            query = [
                cv.imread('Test_images/BBWhite.png',cv.IMREAD_GRAYSCALE),
                cv.imread('Test_images/PlateWhite1.png',cv.IMREAD_GRAYSCALE), 
                cv.imread('Test_images/VoltWhite.png',cv.IMREAD_GRAYSCALE), 
                cv.imread('Test_images/ThinPlate1.png',cv.IMREAD_GRAYSCALE),
                cv.imread('Test_images/Current1.png',cv.IMREAD_GRAYSCALE),
                cv.imread('Test_images/Standoff1.png',cv.IMREAD_GRAYSCALE)
            ]

            #train images // to be replaced by camera feed
            train = [
                frameCV
            ]

            # Initiate SIFT detector
            sift = cv.SIFT_create()

            qkp = []
            qdes = []
            for n in range(len(query)):
                qkp.append([])
                qdes.append([])
                qkp[n], qdes[n] = sift.detectAndCompute(query[n],None)

            tkp = []
            tdes = []
            for n in range(len(train)):
                tkp.append([])
                tdes.append([])
                tkp[n], tdes[n] = sift.detectAndCompute(train[n],None)

            matches = []
            bf = cv.BFMatcher()
            for i in range(len(query)):
                for j in range(len(train)):
                    matches.append([])
                    matches[i].append([])
                    if qdes[i] is not None and tdes[j] is not None:
                        matches[i][j] = bf.knnMatch(qdes[i],tdes[j],k=2)
                    else:
                        matches[i][j] = []

            good = []
            for i in range(len(query)):
                for j in range(len(train)):
                    good.append([])
                    good[i].append([])
                    for m,n in matches[i][j]:
                        if m.distance < 0.6*n.distance:
                            good[i][j].append([m])

            # drawing all matches for each query image with all train images
            imgResults = []
            for i in range(len(query)):
                for j in range(len(train)):
                    imgResults.append([])
                    imgResults[i].append([])
                    imgResults[i][j] = cv.drawMatchesKnn(query[i],qkp[i],train[j],tkp[j],good[i][j],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            image = imgResults[0][0]
            cv.imshow('Feature Matching', image)

            if cv.waitKey(1) == ord('q'):
                break


