import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
# Asynchronous grab
import time
from sympy import Line
from vmbpy import *
# Data collection
import pandas as pd

MIN_INLIERS = 12      # Define this next to MIN_MATCH_COUNT; must be well above 4
MIN_AREA_RATIO = 0.25 # Projected box area vs. original query box area
MAX_AREA_RATIO = 4.0
MIN_MATCH_COUNT = 10 # Minimum number of good matches before proceeding
SPP = 2 # Second(s) per picture

show_process = True # Set to True to show step by step how the program works
live_camera = False # Set to True to use live camera feed, False to use test images
check_angle = False # Set to True to check angle of component relative to ArUco marker
ArUco_images = False # Set to True use the ArUco train images
display_at_end = False # Set to True to display result image at the end of loop
max_loop = 1 # Max number of loops before program ends

#
# Values to be altered for testing
#
sheet_name = "BR_FLANN_Angle"
image_limit = 0 # Set to 0 to load all images in folder
Lowes_ratio = 0.75 # Lowe's ratio test (distance ratio to eliminate false matches), lower = more strict, higher = more lenient
Match_num = 6 # Max number of same type components in a picture
matches_k_value = 2 # Number of matches to compare against for Lowe's ratio test

# STARTOF AI assisted coding - good parameters to alter to improve performance of 
# feature finders and matchers used in query_init and matcher_func 
checks_number = 50 # Number of checks for FLANN matcher [Default = 50, higher = more accurate but slower]
nfeatures_value = 1000 # Number of features to detect for ORB [Default = 500, higher = more accurate but slower]
WTA_K_value = 3 # ORB parameter [Default = 2, higher = more accurate but slower]
# ENDOF AI assisted coding
HessThreshold = 400 # Hessian threshold for SURF [Default = 400]

matcher_option = "FLANN" # Feature matcher to be used
""" 
Matcher Options: 
    "Brute Force"
    "FLANN"
"""

detector_option = "BRISK" # Feature detector to be used
""" 
Detector Options: 
    "SIFT"
    "SURF"
    "ORB"
    "AKAZE"
    "BRISK" 
"""
# Set up ArUco detector
aruco_dict = cv.aruco.getPredefinedDictionary(
    cv.aruco.DICT_4X4_50
)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

# Initiates query images, initiates feature detector and finds the features of query images
#       Returns qkp, qdes, query, detector (keypoints and descriptors, vector of images, detector function)
#
# Sources:
# Feature detection and matching =
#   https://docs.opencv.org/3.4.20/db/d27/tutorial_py_table_of_contents_feature2d.html
#   https://docs.opencv.org/3.4.20/d9/d97/tutorial_table_of_content_features2d.html
def query_init():

    #query image path, component name, upper threshold, lower threshold
    if ArUco_images == True:
        query = [
            [cv.imread('Test_images/ArUco/Query/Pyros.png',cv.IMREAD_GRAYSCALE), "Pyros", 200, 70],
            [cv.imread('Test_images/ArUco/Query/STEV.png',cv.IMREAD_GRAYSCALE), "STEV", 200, 70],
            [cv.imread('Test_images/ArUco/Query/BB79.png',cv.IMREAD_GRAYSCALE), "BB79", 160, 0],
            [cv.imread('Test_images/ArUco/Query/BB97.png',cv.IMREAD_GRAYSCALE), "BB97", 155, 0],
            [cv.imread('Test_images/ArUco/Query/BB84.png',cv.IMREAD_GRAYSCALE), "BB84", 155, 0],
            [cv.imread('Test_images/ArUco/Query/BB96.png',cv.IMREAD_GRAYSCALE), "BB96", 169, 0],
            [cv.imread('Test_images/ArUco/Query/BB82.png',cv.IMREAD_GRAYSCALE), "BB82", 155, 70],
            [cv.imread('Test_images/ArUco/Query/Standoff.png',cv.IMREAD_GRAYSCALE), "Standoff", 255, 95],
            [cv.imread('Test_images/ArUco/Query/CurrShunt.png',cv.IMREAD_GRAYSCALE), "Current Shunt", 160, 110],
            [cv.imread('Test_images/ArUco/Query/BreakResistor.png',cv.IMREAD_GRAYSCALE), "Break Resistor", 165, 70],
            [cv.imread('Test_images/ArUco/Query/BB102.png',cv.IMREAD_GRAYSCALE), "BB102", 165, 70]
        ]
    else:
        query = [
            [cv.imread('Test_images/NonArUco/Query/Pyros.png',cv.IMREAD_GRAYSCALE), "Pyros", 200, 70],
            [cv.imread('Test_images/NonArUco/Query/STEV.png',cv.IMREAD_GRAYSCALE), "STEV", 200, 70],
            [cv.imread('Test_images/NonArUco/Query/BB79.png',cv.IMREAD_GRAYSCALE), "BB79", 142, 0],
            [cv.imread('Test_images/NonArUco/Query/BB97.png',cv.IMREAD_GRAYSCALE), "BB97", 142, 0],
            [cv.imread('Test_images/NonArUco/Query/BB84.png',cv.IMREAD_GRAYSCALE), "BB84", 142, 0],
            [cv.imread('Test_images/NonArUco/Query/BB96.png',cv.IMREAD_GRAYSCALE), "BB96", 142, 0],
            [cv.imread('Test_images/NonArUco/Query/BB61.png',cv.IMREAD_GRAYSCALE), "BB61", 155, 70],
            [cv.imread('Test_images/NonArUco/Query/BB82.png',cv.IMREAD_GRAYSCALE), "BB82", 155, 70],
            [cv.imread('Test_images/NonArUco/Query/Standoff.png',cv.IMREAD_GRAYSCALE), "Standoff", 155, 95],
            [cv.imread('Test_images/NonArUco/Query/CurrShunt.png',cv.IMREAD_GRAYSCALE), "Current Shunt", 155, 103],
            [cv.imread('Test_images/NonArUco/Query/BreakResistor.png',cv.IMREAD_GRAYSCALE), "Break Resistor", 155, 70],
            [cv.imread('Test_images/NonArUco/Query/BB102.png',cv.IMREAD_GRAYSCALE), "BB102", 155, 70]
        ]

    #init feature detector and find features of query images
    if detector_option == "SIFT":
        detector = cv.SIFT_create()
    elif detector_option == "SURF":
        detector = cv.xfeatures2d.SURF_create(HessThreshold)
    elif detector_option == "ORB":
        detector = cv.ORB_create(nfeatures=nfeatures_value, WTA_K = WTA_K_value)
    elif detector_option == "AKAZE":
        detector = cv.xfeatures2d.AKAZE_create()
    elif detector_option == "BRISK":
        detector = cv.xfeatures2d.BRISK_create()

    qkp = []
    qdes = []
    for n in range(len(query)):
        qkp.append([])
        qdes.append([])
        qkp[n], qdes[n] = detector.detectAndCompute(query[n][0],None)

    if detector_option == "STARnBRIEF":
        return qkp, qdes, query, detector

    else:
        return qkp, qdes, query, detector


# Captures a frame of the camera
#       Returns frameCV (captured frame)
#
# Sources:
# Capture a single frame =
#   https://docs.alliedvision.com/pythonAPIManual.html#:~:text=pixel%20format.-,Frame,-%23
#
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
            __, img_upper = cv.threshold(query[i][0], query[i][2], 255, cv.THRESH_BINARY) #thresholding to create binary image
            __, img_lower = cv.threshold(query[i][0], query[i][3], 255, cv.THRESH_BINARY_INV) #thresholding to create binary image
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
                #cv.drawContours(query[i][0],[box],0,(0,0,255),2)
            else:
                print("No contours found")
    
    return boxp

# Checks features of the query and train images for matches
# Also checks which matches are good based on Lowe's ratio test
#       Returns good (vector containing good matches)
def matcher_func(qdes, detector, train_copy, i):
    matches = []
    good = []
    tkp = []
    tdes = []

    if detector_option == "SIFT" or detector_option == "SURF":
        normType = cv.NORM_L2
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = checks_number)
    else:
        if WTA_K_value > 2:
            normType = cv.NORM_HAMMING2
        else:
            normType = cv.NORM_HAMMING

        # https://forum.opencv.org/t/unsupported-error-when-using-orb-and-cv-flannbasedmatcher/5142/2
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
            table_number = 6, # 12
            key_size = 12,     # 20
            multi_probe_level = 1) #2
        search_params = dict(checks=checks_number)



    if matcher_option == "Brute Force":
        matcher = cv.BFMatcher(normType)
    elif matcher_option == "FLANN":
        matcher = cv.FlannBasedMatcher(index_params, search_params)

    tkp, tdes = detector.detectAndCompute(train_copy,None)


    # STARTOF AI assisted coding - Adding the ability to match multiple of the same
    # type of components. Unsuccesful, but solution of increasing the number of matches
    # does lead to increased accuracy in the resulting bounding boxes, thus kept.

    # Change k=2 to k=5 so a single query feature can match multiple objects
    if qdes is not None and tdes is not None and len(tdes) >= Match_num:
        
        matches = matcher.knnMatch(qdes[i], tdes, k=matches_k_value)
    else:
        matches = []

    # Compare candidate matches against the last match in the group
    for n in matches:
        if len(n) == matches_k_value:
            # Check the top matches against the 5th match (match_group[-1])
            for m in n[:-1]:
                if m.distance < Lowes_ratio * n[-1].distance:
                    good.append([m]) #ENDOF AI assisted coding

    return matches, good, tkp

# Draws the good matches onto the image, 
# then plots them in a grid
def image_results(train, query, matched_boxes, cx, cy):
# Create copies of the train images to draw the bounding
    # boxes of all the components at the end.
    display_image = []
    for j in range(len(train)):
        display_image.append(train[j].copy()) 


    # Checks that matched_boxes and centerpoints are not empty, which would cause a crash
    if len(matched_boxes) != 0 and len(cx) != 0 and len(cy) != 0:
        # Draws bounding boxes on respective train images
        for j in range(len(train)):
            for box in matched_boxes[j]:
                cv.polylines(display_image[j], [np.int32(box)], True, (0, 255, 0), 3, cv.LINE_AA)

            for i in range(len(query)):
                # Draws the center points of each component respectively
                for cntr in range(len(cx[i][j])):
                    if cx[i][j][cntr] is not None:
                        cv.circle(display_image[j], 
                                    (cx[i][j][cntr], 
                                    cy[i][j][cntr]), 20, (255, 0, 0), -1)

        #for i in range(3):
        #    for j in range(2):
        #        plt.subplot(2, 3, i * 2 + j + 1)
        #    plt.imshow(display_image[i * 2 + j], cmap='gray')
        #    plt.axis('off')
               
        # Plot display images on a grid
        for j in range(len(train)):
            plt.subplot(1, len(train), j + 1)
            plt.imshow(display_image[j], cmap='gray')
            plt.axis('off')

    #plt.imshow(display_image, cmap='gray')
    plt.show()

# Finds homography and draws the correspondent bounding box.
# Also plots and displays resulting images on a grid (Makes 
# image_results redundant)
#   Returns matched_boxes, cx, cy (bounding boxes and center points of each component)
#
# Sources: 
# Homography and drawing bounding boxes =
#   https://stackoverflow.com/questions/51606215/how-to-draw-bounding-box-on-best-matches
# Feature matching multiple objects of the same type =
#   https://github.com/prob1995/multi_scale_multi_object_template_matching
#
def homography(qkp, query, train, boxp, qdes, detector):
    imgResults = []
    total_matches = []
    total_good = []
    total_keypoints = []
    matched_boxes = []
    total_angles = []
    cx = []
    cy = []

    # loop through all the query images
    for i in range(len(query)):
        h,w = query[i][0].shape[:2] 
        cx.append([])
        cy.append([])
        total_matches.append([])
        total_good.append([])
        total_keypoints.append([])
        total_angles.append([])

        # loop through all the training images
        for j in range(len(train)):
            imgResults.append([])
            imgResults[i].append([])
            matched_boxes.append([])
            cx[i].append([])
            cy[i].append([])
            total_matches[i].append([])
            total_good[i].append([])
            total_keypoints[i].append([])
            total_angles[i].append([])

            train_copy = train[j].copy() # copy of the train image to draw the bounding box on

            # loop for multiple components of the same type
            for k in range(Match_num):
                matches, good, tkp = matcher_func(qdes,  detector,  train_copy, i)

                if len(good)>=MIN_MATCH_COUNT and boxp[i] is not None:
                    src_pts = np.float32([ qkp[i][m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ tkp[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
                    M, ransac_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

                    if M is None:
                        break # leave loop if there's no valid homography matrix

                    # STARTOF AI assisted coding - Adding minimum  number of inliers to prevent
                    # program from creating a box out of only 4 points. Also added other checks
                    # to prevent faulty boxes.

                    # Only count matches RANSAC considers consistent with the homography
                    n_inliers = int(ransac_mask.sum())
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

                    
                    # Fill area of component in image after finding match, so that
                    # it is not in the way of other components of the same type.
                    matching_img = train[j].copy()
                    inpaint_mask = np.ones_like(matching_img) * 255
                    cv.fillPoly(inpaint_mask, [np.int32(dst)], 0)
                    inpaint_mask = cv.bitwise_not(inpaint_mask)
                    train_copy = cv.inpaint(train_copy, inpaint_mask, 3, cv.INPAINT_TELEA)

                    
                    #
                    # Store data for later analysis
                    #
                    total_matches[i][j].append(len(matches))
                    total_good[i][j].append(len(good))
                    total_keypoints[i][j].append(len(tkp))
                    if check_angle == True:
                        angle_deg = get_angle(dst, train_copy)
                        total_angles[i][j].append(angle_deg)
                    else:
                        total_angles[i][j].append([])
                    
                    Moment = cv.moments(dst)
                    cx[i][j].append(int(Moment['m10']/Moment['m00']))
                    cy[i][j].append(int(Moment['m01']/Moment['m00']))
                    """ 
                    cx/cy[i][j] = array containing all the center points in a train image from a specific query
                    cx/cy[i][j][k] = coordinate of 1 component of a component type in an image
                     """
                    matched_boxes[j].append(dst) #storing the bounding box of the component in the train image

                    # Showing step by step how the program works, should be
                    if show_process == True:
                        matchesMask = [[val] for val in ransac_mask.ravel().tolist()]
                        step_by_step_display(imgResults, query, train, cx, cy, i, j, k, dst, matchesMask, tkp, qkp, w, good)
                    
                else:
                    break # not enough matches, check next image


    return matched_boxes, cx, cy, total_matches, total_good, total_keypoints, total_angles

# Show step by step how the program works, drawing the matches
# that fall within the homography and drawing the bounding box 
# of the component
# 
# Sources: 
# Feature matching multiple objects of the same type =
#   https://github.com/prob1995/multi_scale_multi_object_template_matching
# Finding the angle between two lines =
#   https://www.geeksforgeeks.org/python/python-sympy-line-angle_between-method/
#
def step_by_step_display(imgResults, query, train, cx, cy, i, j, k, dst, matchesMask, tkp, qkp, w, good):
    draw_dst = dst + (w, 0)  # adding offset 
    draw_cx = cx[i][j][k] + w # adding offset

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

    # Draw connecting lines then draw bounding box in red
    imgResults[i][j] = cv.drawMatchesKnn(query[i][0],qkp[i],train[j],tkp,good, None,**draw_params)
    imgResults[i][j] = cv.polylines(imgResults[i][j], [np.int32(draw_dst)], True, (0,0,255),3, cv.LINE_AA)
    pt1 = (int(draw_dst[0][0][0]), int(draw_dst[0][0][1]))
    pt2 = (int(draw_dst[1][0][0]), int(draw_dst[1][0][1]))
    cv.line(imgResults[i][j],pt1,pt2,(255,0,0),5)

    cv.circle(imgResults[i][j], (int(draw_cx), int(cy[i][j][k])), 20, (255, 0, 0), -1)
    plt.imshow(imgResults[i][j], cmap='gray')
    plt.show()

# Detects ArUco marker on the image and extracts its y-axis line
# to compare against the detected components
#   Returns angle_deg (angle of component relative to marker)
# 
# Sources:
# Detecting ArUco markers = 
#   https://www.geeksforgeeks.org/computer-vision/detecting-aruco-markers-with-opencv-and-python-1/
#
def get_angle(dst, train_copy):
    angle_deg = []
    
    # Get points for the side of a component
    pt1 = (int(dst[0][0][0]), int(dst[0][0][1]))
    pt2 = (int(dst[1][0][0]), int(dst[1][0][1]))

    corners, ids, rejected = detector.detectMarkers(train_copy)
    if ids is not None:
        # get y-axis of ArUco marker
        pt1_aruco = (int(corners[0][0][0][0]), int(corners[0][0][0][1]))
        pty_aruco = (int(corners[0][0][3][0]), int(corners[0][0][3][1]))
        cv.line(train_copy,pt1_aruco,pty_aruco,(255,0,0),5)

        ptx_aruco = (int(corners[0][0][1][0]), int(corners[0][0][1][1]))
        cv.line(train_copy,pt1_aruco,ptx_aruco,(255,0,0),5)

        l1 = Line(pt1_aruco, pty_aruco)
        l2 = Line(pt1, pt2)
        l3 = Line(pt1_aruco, ptx_aruco)

        angle_rad_y = l2.angle_between(l1)
        angle_deg_y = math.degrees(angle_rad_y)

        angle_rad_x = l2.angle_between(l3)
        angle_deg_x = math.degrees(angle_rad_x)


        # Uses the axes that the side of the component is closest 
        # to - This is because in actual application rotation
        # differences should be very minimal, and a big difference
        # (45+ deg) would only be caused by the side of the component
        # being used for comparison not being in the y-axis
        diffY = abs(pt1[0] - pt2[0])
        diffX = abs(pt1[1] - pt2[1])
        if diffY < diffX:
            # Check which direction the line is pointing, positive is anticlockwise
            if pt1[0] < pt2[0]: 
                angle_deg = 180 - angle_deg_y
            else:
                angle_deg = angle_deg_y - 180

        else:
            # Check which direction the line is pointing, positive is anticlockwise
            if pt2[1] < pt1[1]: 
                angle_deg = - angle_deg_x
            else:
                angle_deg = angle_deg_x

    return angle_deg

# Load images from the specified folder
#   Returns train and train_name (train images and file name)
#
# Sources:
# Looking through all images in a folder =
#   https://www.youtube.com/watch?v=XOWKKUyjhcU
def load_images(root):
    train = []
    train_name = []
    numberOfImages = []
    
    fnames = sorted(os.listdir(root))

    # check if image limit is on
    if image_limit != 0:
        numberOfImages = image_limit
    else:
        numberOfImages = len(fnames)

    for i in range(numberOfImages):
        filepath = os.path.join(root,fnames[i])

        # STARTOF AI assisted coding - How to skip folder
        if not os.path.isfile(filepath):
            continue  # skip subfolders
        # ENDOF AI assisted coding

        train.append(cv.imread(filepath, cv.IMREAD_GRAYSCALE))
        train_name.append(fnames[i]) 

    return train, train_name

# AI generated function - Formats data into  columns and rows
#   Returns rows (variable containing all the data in each row)
def collect_rows(loop_number, query, train_name, total_keypoints, total_matches,
                 total_good, total_angles, loop_time, run_time):
    rows = []
    for i in range(len(query)):
        for j in range(len(train_name)):
            for k in range(len(total_keypoints[i][j])):
                rows.append({
                    "loop": loop_number,
                    "query": query[i][1],
                    "train": train_name[j],
                    "component": k,
                    "keypoints": total_keypoints[i][j][k],
                    "matches": total_matches[i][j][k],
                    "good_matches": total_good[i][j][k],
                    "angle": total_angles[i][j][k],
                    "loop_time_s": loop_time,
                    "total_run_time_s": run_time,
                })
    return rows

# AI generated function - Export data to Excel file
def save_run_to_excel(rows, sheet_name, settings=None, path="Data.xlsx"):
    df = pd.DataFrame(rows)

    # Record the variables you changed as columns, so every sheet documents itself
    if settings:
        for key, value in settings.items():
            df[key] = value

    sheet_name = sheet_name[:31]  # Excel limits sheet names to 31 characters

    if os.path.exists(path):
        with pd.ExcelWriter(path, engine="openpyxl", mode="a",
                            if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# Main
def main():
    all_rows = []
    loop_number = 0
    test_start = time.perf_counter()
    qkp, qdes, query, detector = query_init()

    # get contours and bounding boxes of query images
    boxp = contour_n_bounding(query)

    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()

        with cams[0] as cam:
            frameCV = cam_image(cam)
            start = time.time()

            while True:
                loop_start = time.perf_counter()
                end = time.time()

                # timer, new image every x seconds - defined by SPP
                if end - start > SPP:
                    frameCV = cam_image(cam)
                    start = time.time()
                    #cv.destroyAllWindows()

                # train images / camera feed
                if (live_camera == True):
                    #train = [frameCV]
                    train = [
                        cv.imread('Test_images/NonArUco/Train/Multi1.png',cv.IMREAD_GRAYSCALE),
                        cv.imread('Test_images/NonArUco/Train/Multi2.png',cv.IMREAD_GRAYSCALE),
                        cv.imread('Test_images/NonArUco/Train/Multi3.png',cv.IMREAD_GRAYSCALE),
                        cv.imread('Test_images/NonArUco/Train/Multi4.png',cv.IMREAD_GRAYSCALE),
                    ]
                    train_name = [
                        "Multi1", "Multi2", "Multi3", "Multi4"
                    ]

                else:
                    # Load ArUco images
                    if ArUco_images == True:
                        if check_angle == True:
                            # The 'r' in 'root' and 'sorted' in front of '(os.listdir(root))'
                            # was thanks to the help of AI
                            root = r"Test_images\ArUco\Train\Angle"
                        else:
                            root = r"Test_images\ArUco\Train"
                    
                    # Load non ArUco images        
                    else:
                        if check_angle == True:
                            root = r"Test_images\NonArUco\Train\Angle"
                        else:
                            root = r"Test_images\NonArUco\Train"

                    train, train_name = load_images(root)


                # find homography, and draw respective boxes
                matched_boxes, cx, cy, total_matches, total_good, total_keypoints, total_angles = homography( qkp, query, train, boxp, qdes, detector)

                # End of performance timer, and print of all test results
                # (Taken before image_results however the time that function takes to run should be consistent)
                test_end = time.perf_counter()
                loop_number += 1

                # STARTOF AI generated code - A way to export all the test data being
                # saved into excel so that it can be analysed
                all_rows.extend(collect_rows(loop_number, query, train_name, total_keypoints, total_matches,
                 total_good, total_angles, test_end - loop_start, test_end - test_start))
                # ENDOF AI generated code

                # display images on a grid
                if display_at_end == True:
                    image_results(train, query, matched_boxes, cx, cy)

                if loop_number >= max_loop:   # or whatever condition ends your test
                    break

                if cv.waitKey(1) == ord('q'):
                    break

            save_run_to_excel(
                all_rows,
                sheet_name=sheet_name,
                settings={"Lowes_ratio": Lowes_ratio,
                        "MIN_MATCH_COUNT": MIN_MATCH_COUNT, "Match_num": Match_num, "image_limit": image_limit},
            )


if __name__ == '__main__':
    main()
