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

""" 
[cv.imread('Test_images/Query/Pyros.png',cv.IMREAD_GRAYSCALE), "Pyros"],
[cv.imread('Test_images/Query/STEV.png',cv.IMREAD_GRAYSCALE), "STEV"],
[cv.imread('Test_images/Query/BB79.png',cv.IMREAD_GRAYSCALE), "BB79"],
[cv.imread('Test_images/Query/BB97.png',cv.IMREAD_GRAYSCALE), "BB97"],
[cv.imread('Test_images/Query/BB84.png',cv.IMREAD_GRAYSCALE), "BB84"],
[cv.imread('Test_images/Query/BB96.png',cv.IMREAD_GRAYSCALE), "BB96"],
[cv.imread('Test_images/Query/BB61.png',cv.IMREAD_GRAYSCALE), "BB61"],
[cv.imread('Test_images/Query/BB82.png',cv.IMREAD_GRAYSCALE), "BB82"],
[cv.imread('Test_images/Query/Standoff.png',cv.IMREAD_GRAYSCALE), "Standoff"],
[cv.imread('Test_images/Query/CurrShunt.png',cv.IMREAD_GRAYSCALE), "Current Shunt"],
[cv.imread('Test_images/Query/BreakResistor.png',cv.IMREAD_GRAYSCALE), "Break Resistor"],
[cv.imread('Test_images/Query/BB102.png',cv.IMREAD_GRAYSCALE), "BB102"] 
"""

img1 = cv.imread('Test_images/Query/BB102.png',cv.IMREAD_GRAYSCALE)   # queryImage

#Pyro and STEVS
#__, img_upper = cv.threshold(img1, 200, 255, cv.THRESH_BINARY) 
#__, img_lower = cv.threshold(img1, 70, 255, cv.THRESH_BINARY_INV) 

#BB79, BB97, BB84, BB96
#__, img_upper = cv.threshold(img1, 142, 255, cv.THRESH_BINARY) 
#__, img_lower = cv.threshold(img1, 0, 255, cv.THRESH_BINARY_INV) 

#BB61, BB82, Break Resistor
#__, img_upper = cv.threshold(img1, 155, 255, cv.THRESH_BINARY)
#__, img_lower = cv.threshold(img1, 70, 255, cv.THRESH_BINARY_INV)
 
#Standoff
#__, img_upper = cv.threshold(img1, 155, 255, cv.THRESH_BINARY) 
#__, img_lower = cv.threshold(img1, 95, 255, cv.THRESH_BINARY_INV) 

#Current Shunt
#__, img_upper = cv.threshold(img1, 155, 255, cv.THRESH_BINARY) 
#__, img_lower = cv.threshold(img1, 103, 255, cv.THRESH_BINARY_INV) 

#####################################
# Contour detection and bounding box
#####################################
__, img_upper = cv.threshold(img1, 155, 255, cv.THRESH_BINARY) #thresholding to create binary image
__, img_lower = cv.threshold(img1, 70, 255, cv.THRESH_BINARY_INV) #thresholding to create binary image
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

