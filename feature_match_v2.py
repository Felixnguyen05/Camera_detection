import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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
    cv.imread('Test_images/Plates1.png',cv.IMREAD_GRAYSCALE), 
    cv.imread('Test_images/WhiteA6.png',cv.IMREAD_GRAYSCALE), 
    cv.imread('Test_images/WhiteA1.png',cv.IMREAD_GRAYSCALE),
    cv.imread('Test_images/WhiteA2.png',cv.IMREAD_GRAYSCALE),
    cv.imread('Test_images/WhiteA3.png',cv.IMREAD_GRAYSCALE),
    cv.imread('Test_images/WhiteA4.png',cv.IMREAD_GRAYSCALE),
    cv.imread('Test_images/WhiteA5.png',cv.IMREAD_GRAYSCALE),
    cv.imread('Test_images/PlateWhite2.png',cv.IMREAD_GRAYSCALE),
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
        matches[i][j] = bf.knnMatch(qdes[i],tdes[j],k=2)

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

#plt.figure()
#plt.subplot(6,3,1)
#plt.axis('off')

# plotting all result images together in a single figure
for n in range(len(query)):
    for j in range(3):
        plt.subplot(6, 3, n * 3 + j + 1)
        plt.imshow(imgResults[n][j])
        plt.axis('off')

#plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.1)
plt.show()

