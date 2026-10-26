import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# Asynchronous grab
import time
from vmbpy import *

MIN_MATCH_COUNT = 10

with VmbSystem.get_instance() as vmb:
    cams = vmb.get_all_cameras()
    with cams[0] as cam:

        frame = cam.get_frame()
        frame_mono8 = frame.convert_pixel_format(PixelFormat.Mono8)
        frameCV = frame_mono8.as_opencv_image()
        start = time.time()
        while True: 
            end = time.time()
            if end - start > 5:
                frame = cam.get_frame()
                frame_mono8 = frame.convert_pixel_format(PixelFormat.Mono8)
                frameCV = frame_mono8.as_opencv_image()
                plt.close('all')
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
                cv.imread('Test_images/Plates1.png',cv.IMREAD_GRAYSCALE)
            ]

            #####################################
            # Contour detection and bounding box
            #####################################
            boxp = []
            for i in range(len(query)):
                for j in range(len(train)):
                    __, img_upper = cv.threshold(query[i], 200, 255, cv.THRESH_BINARY) #thresholding to create binary image
                    __, img_lower = cv.threshold(query[i], 110, 255, cv.THRESH_BINARY_INV) #thresholding to create binary image
                    img_mask = cv.bitwise_or(img_upper, img_lower)

                    img_masked = cv.bitwise_and(query[i], query[i], mask=img_mask)

                    comp_contours, _ = cv.findContours(img_masked, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                    boxp.append([])
                    boxp[i].append([])
                    if comp_contours:
                        cnt = max(comp_contours, key=cv.contourArea)

                        rect = cv.minAreaRect(cnt)
                        boxp[i][j] = cv.boxPoints(rect)
                        box = np.int32(boxp[i][j])
                        cv.drawContours(query[i],[box],0,(0,0,255),2)
                    else:
                        print("No contours found")

            ###################################
            # Feature matching
            ###################################
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

            ############################################################
            # Homography and drawing of bounding boxes on train images
            ############################################################
            imgResults = []
            for i in range(len(query)):
                for j in range(len(train)):

                    imgResults.append([])
                    imgResults[i].append([])

                    if len(good[i][j])>=MIN_MATCH_COUNT:
                        src_pts = np.float32([ qkp[i][m[0].queryIdx].pt for m in good[i][j] ]).reshape(-1,1,2)
                        dst_pts = np.float32([ tkp[j][m[0].trainIdx].pt for m in good[i][j] ]).reshape(-1,1,2)
                        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
                        matchesMask = [[val] for val in mask.ravel().tolist()]
                        h,w = query[i].shape[:2]
                        pts = np.float32(boxp[i][j]).reshape(-1,1,2)

                        dst = cv.perspectiveTransform(pts,M)
                        dst += (w, 0)  # adding offset TRY TO UNDERSTAND THIS

                        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                                    singlePointColor = None,
                                    matchesMask = matchesMask, # draw only inliers
                                    flags = 2)
                    
                        # Draw matches
                        imgResults[i][j] = cv.drawMatchesKnn(query[i],qkp[i],train[j],tkp[j],good[i][j],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                        # Draw bounding box in Red
                        imgResults[i][j] = cv.polylines(imgResults[i][j],[np.int32(dst)],True,(0,0,255),3, cv.LINE_AA)
                    else:
                        print( "Not enough matches found")

            ################################################################
            # Plotting all results in a grid
            ################################################################
            #for n in range(len(query)):
            #    for j in range(len(train)):
            #        plt.subplot(len(query), len(train), n * len(train) + j + 1)
            #        plt.imshow(imgResults[n][j])
            #        plt.axis('off')

            #plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.1)
            #plt.show()    

            if cv.waitKey(1) == ord('q'):
                break








