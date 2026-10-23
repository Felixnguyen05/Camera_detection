import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


#query = [
#    cv.imread('Test_images/BBWhite.png',cv.IMREAD_GRAYSCALE),
#    cv.imread('Test_images/PlateWhite1.png',cv.IMREAD_GRAYSCALE), 
#    cv.imread('Test_images/VoltWhite.png',cv.IMREAD_GRAYSCALE), 
#    cv.imread('Test_images/ThinPlate1.png',cv.IMREAD_GRAYSCALE),
#    cv.imread('Test_images/Current1.png',cv.IMREAD_GRAYSCALE),
#    cv.imread('Test_images/Standoff1.png',cv.IMREAD_GRAYSCALE)
#]

img1 = cv.imread('Test_images/BBWhite.png',cv.IMREAD_GRAYSCALE)          # queryImage

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


plt.subplot(5,1,1),plt.imshow(img1, cmap='gray'), plt.axis('off')
plt.subplot(5,1,2),plt.imshow(img_upper, cmap='gray'), plt.axis('off')
plt.subplot(5,1,3),plt.imshow(img_lower, cmap='gray'), plt.axis('off')
plt.subplot(5,1,4),plt.imshow(img_mask, cmap='gray'), plt.axis('off')
plt.subplot(5,1,5),plt.imshow(img1_masked, cmap='gray'), plt.axis('off')
plt.show()

