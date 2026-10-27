import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# Asynchronous grab
import time
from vmbpy import *

MIN_MATCH_COUNT = 10
Lowes_ratio = 0.6 # Test changing the ratio?
SPP = 2 # Second(s) per picture

# Initiates query images, initiates sift and finds the features of query images
#       Returns qkp, qdes, query, sift (features, vector of images, sift function)
def query_init():
    #query images
    query = [
        cv.imread('Test_images/BBWhite.png',cv.IMREAD_GRAYSCALE),
        cv.imread('Test_images/PlateWhite1.png',cv.IMREAD_GRAYSCALE), 
        cv.imread('Test_images/VoltWhite.png',cv.IMREAD_GRAYSCALE), 
        cv.imread('Test_images/ThinPlate1.png',cv.IMREAD_GRAYSCALE),
        cv.imread('Test_images/Current1.png',cv.IMREAD_GRAYSCALE),
        cv.imread('Test_images/Standoff1.png',cv.IMREAD_GRAYSCALE)
    ]

    #init sift and find features of query images
    sift = cv.SIFT_create()

    qkp = []
    qdes = []
    for n in range(len(query)):
        qkp.append([])
        qdes.append([])
        qkp[n], qdes[n] = sift.detectAndCompute(query[n],None)

    return qkp, qdes, query, sift

# Captures a frame of the camera
#       Returns frameCV (captured frame)
def cam_image(cam):
    frame = cam.get_frame()
    frame_mono8 = frame.convert_pixel_format(PixelFormat.Mono8)
    frameCV = frame_mono8.as_opencv_image()
    return frameCV

# Creates light level thresholds to mask out components so that a minimum
# area rectangle can be drawn. Add this function before while loop (perhaps
# merge with query_init)
#       Returns boxp (vector containing the 4 points to make the rectangles)
def contour_n_bounding(train, query):
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
    
    return boxp

# Checks features of the query and train images for matches
# Also checks which matches are good based on Lowe's ratio test
#       Returns good (vector containing good matches)
def matcher(train_var, query, qdes, tdes):
    matches = []
    good = []
    bf = cv.BFMatcher()

    for i in range(len(query)):
        for j in range(len(train_var)):
            matches.append([])
            matches[i].append([])

            if qdes[i] is not None and tdes[j] is not None:
                matches[i][j] = bf.knnMatch(qdes[i],tdes[j],k=2)
            else:
                matches[i][j] = []

            # checking which matches are good based on Lowe's ratio test
            good.append([])
            good[i].append([])
            for m,n in matches[i][j]:
                if m.distance < Lowes_ratio*n.distance:
                    good[i][j].append([m])
    return good

# Draws the good matches onto the image, 
# then plots them in a grid
def image_results(train, query, qkp, tkp, good):
    imgResults = []

    for i in range(len(query)):
        for j in range(len(train)):
            imgResults.append([])
            imgResults[i].append([])
            imgResults[i][j] = cv.drawMatchesKnn(query[i],qkp[i],train[j],tkp[j],good[i][j],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            #
            #Plotting all results in a grid
            #
            plt.subplot(len(query), len(train), i * len(train) + j + 1)
            plt.imshow(imgResults[i][j], cmap='gray')
            plt.axis('off')

# Finds homography and draws the correspondent bounding box.
# Also plots and displays resulting images on a grid (Makes 
# image_results redundant)
#           (Figure out what src_pts, pts, and dst do!)
def homography(good, qkp, tkp, query, train, boxp):
    imgResults = []
    h,w = query[0].shape[:2]
    
    for i in range(len(query)):
        for j in range(len(train)):
            imgResults.append([])
            imgResults[i].append([])

            if len(good[i][j])>=MIN_MATCH_COUNT and boxp[i][j] is not None:
                src_pts = np.float32([ qkp[i][m[0].queryIdx].pt for m in good[i][j] ]).reshape(-1,1,2)
                dst_pts = np.float32([ tkp[j][m[0].trainIdx].pt for m in good[i][j] ]).reshape(-1,1,2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

                # Check whether a valid homography matrix was found
                if M is not None:
                    matchesMask = [[val] for val in mask.ravel().tolist()]
                    pts = np.float32(boxp[i][j]).reshape(-1,1,2)

                    dst = cv.perspectiveTransform(pts,M) 
                    dst += (w, 0)  # adding offset 

                    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                                    singlePointColor = None,
                                    matchesMask = matchesMask, # draw only inliers
                                    flags = 2)

                    # Draw connecting lines then draw bounding box in red
                    imgResults[i][j] = cv.drawMatchesKnn(query[i],qkp[i],train[j],tkp[j],good[i][j], None,**draw_params)
                    imgResults[i][j] = cv.polylines(imgResults[i][j], [np.int32(dst)], True, (0,0,255),3, cv.LINE_AA)
                    
                else:
                    # draw images without connections side-by-side to prevent crashing
                    imgResults[i][j] = cv.drawMatchesKnn(query[i], qkp[i], train[j], tkp[j], [], None, flags=2)
            else:
                print( "Not enough matches found OR boxp is empty")

                # draw images without connections side-by-side to prevent crashing
                imgResults[i][j] = cv.drawMatchesKnn(query[i], qkp[i], train[j], tkp[j], [], None, flags=2)

            plt.subplot(len(query), len(train), i * len(train) + j + 1)
            plt.imshow(imgResults[i][j], cmap='gray')
            plt.axis('off')

# Main
def main():
    qkp, qdes, query, sift = query_init()

    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()

        with cams[0] as cam:
            frameCV = cam_image(cam)
            start = time.time()

            while True:
                end = time.time()

                # timer, new image every x seconds - defined by SPP
                if end - start > SPP:
                    frameCV = cam_image(cam)
                    start = time.time()

                # train images / camera feed
                # and feature finding of these images
                train = [
                    #frameCV,
                    cv.imread('Test_images/Plates1.png',cv.IMREAD_GRAYSCALE), 
                    cv.imread('Test_images/WhiteA6.png',cv.IMREAD_GRAYSCALE), 
                    cv.imread('Test_images/WhiteA1.png',cv.IMREAD_GRAYSCALE)
                ]

                # get contours and bounding boxes of query images
                boxp = contour_n_bounding(train, query)

                tkp = []
                tdes = []
                for n in range(len(train)):
                    tkp.append([])
                    tdes.append([])
                    tkp[n], tdes[n] = sift.detectAndCompute(train[n],None)

                # find good matches
                good = matcher(train, query, qdes, tdes)

                # find homography, and draw respective boxes
                homography(good, qkp, tkp, query, train, boxp)
                plt.show()    
            
                #image = imgResults[0][0]
                #cv.imshow('Feature Matching', image)
                
                if cv.waitKey(1) == ord('q'):
                    break


if __name__ == '__main__':
    main()
