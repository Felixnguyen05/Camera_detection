import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

#query = [
#    cv.imread('Test_images/BBWhite.png',cv.IMREAD_GRAYSCALE),
#    cv.imread('Test_images/PlateWhite1.png',cv.IMREAD_GRAYSCALE), 
#    cv.imread('Test_images/VoltWhite.png',cv.IMREAD_GRAYSCALE), 
#    cv.imread('Test_images/ThinPlate1.png',cv.IMREAD_GRAYSCALE),
#    cv.imread('Test_images/Current1.png',cv.IMREAD_GRAYSCALE),
#    cv.imread('Test_images/Standoff1.png',cv.IMREAD_GRAYSCALE)
#]

img1 = cv.imread('Test_images/PlateWhite1.png',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('Test_images/Plates1.png',cv.IMREAD_GRAYSCALE) # trainImage


#####################################
# Contour detection and bounding box
#####################################
__, img_upper = cv.threshold(img1, 200, 255, cv.THRESH_BINARY) #thresholding to create binary image
__, img_lower = cv.threshold(img1, 110, 255, cv.THRESH_BINARY_INV) #thresholding to create binary image
img_mask = cv.bitwise_or(img_upper, img_lower)

img1_masked = cv.bitwise_and(img1, img1, mask=img_mask)

comp_contours, _ = cv.findContours(img1_masked, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

cnt = []
rect = []
box = []
boxp = []
if comp_contours:
    cnt = max(comp_contours, key=cv.contourArea)

    rect = cv.minAreaRect(cnt)
    boxp = cv.boxPoints(rect)
    box = np.int32(boxp)
    cv.drawContours(img1,[box],0,(0,0,255),2)
else:
    print("No contours found")

#################################
# Feature matching and homography
#################################

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv.BFMatcher() #Flann matcher might be better for live use
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.6*n.distance: #Test changing the ratio to see if results improve
        good.append([m])

###################################################
# Different from feature_match_v1 after this point
###################################################
# try to understand what any of this does

img_result = img2
if len(good)>=MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2) #try to understand what this does
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = [[val] for val in mask.ravel().tolist()]
    h,w = img1.shape[:2]
    pts = np.float32(boxp).reshape(-1,1,2)

    dst = cv.perspectiveTransform(pts,M)
    dst += (w, 0)  # adding offset

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                singlePointColor = None,
                matchesMask = matchesMask, # draw only inliers
                flags = 2)

    img_result = cv.drawMatchesKnn(img1,kp1,img2,kp2,good, None,**draw_params)

    # Draw bounding box in Red
    img_result = cv.polylines(img_result, [np.int32(dst)], True, (0,0,255),3, cv.LINE_AA)
else:
    print( "Not enough matches found")

plt.imshow(img_result, cmap='gray'), plt.axis('off')

plt.show()

