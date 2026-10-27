import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# Asynchronous grab
import time
from vmbpy import *

MIN_MATCH_COUNT = 10
Lowes_ratio = 0.6 # Test changing the ratio?
SPP = 2 # Second(s) per picture
Match_num = 5 # Max number of same type components in a picture

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
def contour_n_bounding(query):
    boxp = []
    for i in range(len(query)):
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
                boxp[i] = cv.boxPoints(rect)
                box = np.int32(boxp[i])
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

            # Change k=2 to k=5 so a single query feature can match multiple objects
            if qdes[i] is not None and tdes[j] is not None and len(tdes[j]) >= Match_num:
                matches[i][j] = bf.knnMatch(qdes[i], tdes[j], k=Match_num)
            else:
                matches[i][j] = []

            # Compare candidate matches against the last match in the group
            good.append([])
            good[i].append([])
            for n in matches[i][j]:
                if len(n) == Match_num:
                    # Check the top matches against the 5th match (match_group[-1])
                    for m in n[:-1]:
                        if m.distance < Lowes_ratio * n[-1].distance:
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
def homography(qkp, tkp, query, train, boxp, qdes, tdes):
    imgResults = []
    matches = []
    good = []
    bf = cv.BFMatcher()

    for i in range(len(query)):
        h,w = query[i].shape[:2]

        for j in range(len(train)):
            imgResults.append([])
            imgResults[i].append([])
            good.append([])
            good[i].append([])
            for k in range(Match_num):

                # Change k so a single query feature can match multiple objects
                if qdes[i] is not None and tdes[j] is not None and len(tdes[j]) >= Match_num:
                    matches = bf.knnMatch(qdes[i], tdes[j], k=Match_num)
                else:
                    matches = []

                # Compare candidate matches against the last match in the group
                for n in matches:
                    if len(n) == Match_num:
                        # Check the top matches against the 5th match (match_group[-1])
                        for m in n[:-1]:
                            if m.distance < Lowes_ratio * n[-1].distance:
                                good[i][j].append([m])

                if len(good[i][j])>=MIN_MATCH_COUNT and boxp[i] is not None:
                    src_pts = np.float32([ qkp[i][m[0].queryIdx].pt for m in good[i][j] ]).reshape(-1,1,2)
                    dst_pts = np.float32([ tkp[j][m[0].trainIdx].pt for m in good[i][j] ]).reshape(-1,1,2)
                    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

                    # Check whether a valid homography matrix was found
                    if M is not None:
                        matchesMask = [[val] for val in mask.ravel().tolist()]
                        pts = np.float32(boxp[i]).reshape(-1,1,2)

                        dst = cv.perspectiveTransform(pts,M) 

                        matching_img = train[j].copy()
                        mask = np.ones_like(matching_img) * 255
                        cv.fillPoly(mask, [np.int32(dst)], 0)
                        mask = cv.bitwise_not(mask)
                        train[j] = cv.inpaint(train[j], mask, 3, cv.INPAINT_TELEA)
                        cv.imshow("masked train", train[j])

                        draw_dst = dst + (w, 0)  # adding offset 

                        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                                        singlePointColor = None,
                                        matchesMask = matchesMask, # draw only inliers
                                        flags = 2)

                        # Draw connecting lines then draw bounding box in red
                        imgResults[i][j] = cv.drawMatchesKnn(query[i],qkp[i],train[j],tkp[j],good[i][j], None,**draw_params)
                        imgResults[i][j] = cv.polylines(imgResults[i][j], [np.int32(draw_dst)], True, (0,0,255),3, cv.LINE_AA)

                    else:
                        # draw images without connections side-by-side to prevent crashing
                        imgResults[i][j] = cv.drawMatchesKnn(query[i], qkp[i], train[j], tkp[j], [], None, flags=2)
                else:
                    print( "Not enough matches found OR boxp is empty")

                    # draw images without connections side-by-side to prevent crashing
                    imgResults[i][j] = cv.drawMatchesKnn(query[i], qkp[i], train[j], tkp[j], [], None, flags=2)

            
    # TILE_W, TILE_H = 500, 250  # Standard size for each grid cell
    TILE_W, TILE_H = round(1530 / len(train)), round(830 / len(query))  # Size dependent on the amount of query and train images
    grid_rows = []

    for i in range(len(query)):
        row_imgs = []
        for j in range(len(train)):
            # Resize each result so all tiles have identical dimensions
            row_imgs.append(cv.resize(imgResults[i][j], (TILE_W, TILE_H)))
            
        # Stack all train columns horizontally for this query row
        grid_rows.append(np.hstack(row_imgs))

    # Stack all query rows vertically into one master grid image
    final_grid = np.vstack(grid_rows)
    cv.imshow("Homography Grid", final_grid)

# Main
def main():
    qkp, qdes, query, sift = query_init()

    # get contours and bounding boxes of query images
    boxp = contour_n_bounding(query)

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
                    #cv.destroyAllWindows()

                # train images / camera feed
                # and feature finding of these images
                train = [
                    cv.imread('Test_images/Multi1.png',cv.IMREAD_GRAYSCALE)
                ]

                tkp = []
                tdes = []
                for n in range(len(train)):
                    tkp.append([])
                    tdes.append([])
                    tkp[n], tdes[n] = sift.detectAndCompute(train[n],None)

                # find good matches
                #good = matcher(train, query, qdes, tdes)

                # find homography, and draw respective boxes
                homography( qkp, tkp, query, train, boxp, qdes, tdes)
                #plt.show()    
            
                #image = imgResults[0][0]
                #cv.imshow('Feature Matching', image)
                
                if cv.waitKey(1) == ord('q'):
                    break


if __name__ == '__main__':
    main()
