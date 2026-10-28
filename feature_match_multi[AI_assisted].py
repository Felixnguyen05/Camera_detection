import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# Asynchronous grab
import time
from vmbpy import *

MIN_INLIERS = 12      # define this next to MIN_MATCH_COUNT; must be well above 4
MIN_AREA_RATIO = 0.25 # projected box area vs. original query box area
MAX_AREA_RATIO = 4.0

MIN_MATCH_COUNT = 10
Lowes_ratio = 0.85 # Test changing the ratio?
SPP = 2 # Second(s) per picture
Match_num = 5 # Max number of same type components in a picture

# Initiates query images, initiates sift and finds the features of query images
#       Returns qkp, qdes, query, sift (features, vector of images, sift function)
def query_init():
    
    #query images and their names
    query = [
        [cv.imread('Test_images/BBWhite.png',cv.IMREAD_GRAYSCALE), "BBWhite"],
        [cv.imread('Test_images/PlateWhite1.png',cv.IMREAD_GRAYSCALE), "PlateWhite"], 
        [cv.imread('Test_images/VoltWhite.png',cv.IMREAD_GRAYSCALE), "VoltWhite"], 
        [cv.imread('Test_images/ThinPlate1.png',cv.IMREAD_GRAYSCALE), "ThinPlate"],
        [cv.imread('Test_images/Current1.png',cv.IMREAD_GRAYSCALE), "Current"],
        [cv.imread('Test_images/Standoff1.png',cv.IMREAD_GRAYSCALE), "Standoff"]
    ]

    #init sift and find features of query images
    sift = cv.SIFT_create()

    qkp = []
    qdes = []
    for n in range(len(query)):
        qkp.append([])
        qdes.append([])
        qkp[n], qdes[n] = sift.detectAndCompute(query[n][0],None)

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
#
# Sources:
# Contours and minimum area rectangles = 
#   https://docs.opencv.org/3.4.20/dd/d49/tutorial_py_contour_features.html
#   https://www.geeksforgeeks.org/python/finding-minimum-enclosing-rectangle-in-opencv-python/
#
def contour_n_bounding(query):
    boxp = []
    
    for i in range(len(query)):
            __, img_upper = cv.threshold(query[i][0], 200, 255, cv.THRESH_BINARY) #thresholding to create binary image
            __, img_lower = cv.threshold(query[i][0], 110, 255, cv.THRESH_BINARY_INV) #thresholding to create binary image
            img_mask = cv.bitwise_or(img_upper, img_lower)

            img_masked = cv.bitwise_and(query[i][0], query[i][0], mask=img_mask)

            comp_contours, _ = cv.findContours(img_masked, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            boxp.append([])
            boxp[i].append([])
            if comp_contours:
                cnt = max(comp_contours, key=cv.contourArea)

                rect = cv.minAreaRect(cnt)
                boxp[i] = cv.boxPoints(rect)
                box = np.int32(boxp[i])
                cv.drawContours(query[i][0],[box],0,(0,0,255),2)
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
            imgResults[i][j] = cv.drawMatchesKnn(query[i][0],qkp[i],train[j],tkp[j],good[i][j],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            #
            #Plotting all results in a grid
            #
            plt.subplot(len(query), len(train), i * len(train) + j + 1)
            plt.imshow(imgResults[i][j], cmap='gray')
            plt.axis('off')

# Finds homography and draws the correspondent bounding box.
# Also plots and displays resulting images on a grid (Makes 
# image_results redundant)
#
# Sources: 
# Homography and drawing bounding boxes =
#   https://stackoverflow.com/questions/51606215/how-to-draw-bounding-box-on-best-matches
# Feature matching on multiple objects =
#   https://github.com/prob1995/multi_scale_multi_object_template_matching
#
def homography(qkp, query, train, boxp, qdes, sift):
    imgResults = []
    matches = []
    good = []
    bf = cv.BFMatcher()
    matched_boxes = []

    tkp = []
    tdes = []

    display_image = train[0].copy()  

    for i in range(len(query)):
        h,w = query[i][0].shape[:2]

        for j in range(len(train)):
            imgResults.append([])
            imgResults[i].append([])
            good.append([])
            good[i].append([])

            for k in range(Match_num):
                good[i][j].clear()
                tkp.append([])
                tdes.append([])
                tkp[j], tdes[j] = sift.detectAndCompute(train[j],None)

                # Change k value to Match_num so a single query feature can match multiple objects
                if qdes[i] is not None and tdes[j] is not None and len(tdes[j]) >= Match_num:
                    matches = bf.knnMatch(qdes[i], tdes[j], k=Match_num)
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
                                good[i][j].append([m]) #ENDOF AI assisted coding

                if len(good[i][j])>=MIN_MATCH_COUNT and boxp[i] is not None:
                    src_pts = np.float32([ qkp[i][m[0].queryIdx].pt for m in good[i][j] ]).reshape(-1,1,2)
                    dst_pts = np.float32([ tkp[j][m[0].trainIdx].pt for m in good[i][j] ]).reshape(-1,1,2)
                    M, ransac_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

                    if M is None:
                        break # leave loop if there's no valid homography matrix

                    # STARTOF AI assisted coding - Adding minimum  number of inliers to prevent
                    # program from creating a box out of only 4 points. Also added other checks
                    # to prevent faulty boxes.

                    # Only count matches RANSAC considers consistent with the homography
                    n_inliers = int(ransac_mask.sum())
                    print(f"query {i} / train {j} / pass {k}: {n_inliers} inliers")
                    if n_inliers < MIN_INLIERS:
                        break   # nothing real left to find, stop searching this train image

                    pts = np.float32(boxp[i]).reshape(-1,1,2)
                    dst = cv.perspectiveTransform(pts, M)

                    # Reject degenerate boxes (skewed, inverted, or wildly wrong size)
                    box_area = cv.contourArea(pts)
                    dst_area = cv.contourArea(dst)
                    if (box_area == 0
                            or not cv.isContourConvex(np.int32(dst)) # checks that shape doesn't cross itself
                            or not (MIN_AREA_RATIO < dst_area / box_area < MAX_AREA_RATIO)): 
                            # dst_area/box_area gives the ratio of the query bounding box to the train bounding box
                            # this should fit between the MIN and MAX established
                        break # ENDOF AI assisted coding 
                    
                    matchesMask = [[val] for val in ransac_mask.ravel().tolist()]
                    pts = np.float32(boxp[i]).reshape(-1,1,2)

                    dst = cv.perspectiveTransform(pts,M) 
                    matched_boxes.append(dst)

                    # Fill in image after finding match, so that it is not in
                    # the way of other components of the same type
                    matching_img = train[j].copy()
                    inpaint_mask = np.ones_like(matching_img) * 255
                    cv.fillPoly(inpaint_mask, [np.int32(dst)], 0)
                    inpaint_mask = cv.bitwise_not(inpaint_mask)
                    train[j] = cv.inpaint(train[j], inpaint_mask, 3, cv.INPAINT_TELEA)
                    cv.imshow("masked train", train[j])

                    draw_dst = dst + (w, 0)  # adding offset 

                    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                                    singlePointColor = None,
                                    matchesMask = matchesMask, # draw only inliers
                                    flags = 2)

                    # Draw connecting lines then draw bounding box in red
                    imgResults[i][j] = cv.drawMatchesKnn(query[i][0],qkp[i],train[j],tkp[j],good[i][j], None,**draw_params)
                    imgResults[i][j] = cv.polylines(imgResults[i][j], [np.int32(draw_dst)], True, (0,0,255),3, cv.LINE_AA)
                    plt.imshow(imgResults[i][j], cmap='gray')
                    plt.show()

                else:
                    break # not enough matches, check next image

    # STARTOF AI assisted coding, displaying images on a grid using opencv
    # instead of matplotlib which is what was used prior.       
    """# TILE_W, TILE_H = 500, 250  # Standard size for each grid cell
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
    cv.imshow("Homography Grid", final_grid)"""
    # ENDOF AI assisted coding

    for box in matched_boxes:
        cv.polylines(display_image, [np.int32(box)], True, (0, 255, 0), 3, cv.LINE_AA)

    plt.imshow(display_image)
    plt.show()

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
                    cv.imread('Test_images/Plates1.png',cv.IMREAD_GRAYSCALE), 
                    cv.imread('Test_images/WhiteA6.png',cv.IMREAD_GRAYSCALE), 
                    cv.imread('Test_images/WhiteA1.png',cv.IMREAD_GRAYSCALE),
                    cv.imread('Test_images/Multi1.png',cv.IMREAD_GRAYSCALE)
                ]

                
                # find good matches
                #good = matcher(train, query, qdes, tdes)

                # find homography, and draw respective boxes
                homography( qkp, query, train, boxp, qdes, sift)
                #plt.show()    
            
                #image = imgResults[0][0]
                #cv.imshow('Feature Matching', image)
                
                if cv.waitKey(1) == ord('q'):
                    break


if __name__ == '__main__':
    main()