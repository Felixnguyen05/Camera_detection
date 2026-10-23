import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('Test_images/BBWhite.png',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('Test_images/WhiteA5.png',cv.IMREAD_GRAYSCALE) # trainImage
img3 = cv.imread('Test_images/WhiteA4.png',cv.IMREAD_GRAYSCALE) # trainImage
img4 = cv.imread('Test_images/WhiteA6.png',cv.IMREAD_GRAYSCALE) # trainImage


# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
kp3, des3 = sift.detectAndCompute(img3,None)
kp4, des4 = sift.detectAndCompute(img4,None)

# BFMatcher with default params
bf = cv.BFMatcher() #Flann matcher might be better for live use
matches2 = bf.knnMatch(des1,des2,k=2)
matches3 = bf.knnMatch(des1,des3,k=2)
matches4 = bf.knnMatch(des1,des4,k=2)

# Apply ratio test
good2 = []
for m,n in matches2:
    if m.distance < 0.6*n.distance: #Test changing the ratio to see if results improve
        good2.append([m])

# Apply ratio test
good3 = []
for m,n in matches3:
    if m.distance < 0.6*n.distance:
        good3.append([m])


# Apply ratio test
good4 = []
for m,n in matches4:
    if m.distance < 0.6*n.distance:
        good4.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
img2A = cv.drawMatchesKnn(img1,kp1,img2,kp2,good2,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img3A = cv.drawMatchesKnn(img1,kp1,img3,kp3,good3,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img4A = cv.drawMatchesKnn(img1,kp1,img4,kp4,good4,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img2A),plt.show()
plt.imshow(img3A),plt.show()
plt.imshow(img4A),plt.show()