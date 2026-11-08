import cv2
import matplotlib.pyplot as plt

image = cv2.imread("Test_images/NonArUco/Train/Angle/BB102HCS5_2.png",cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not loaded. Please check the image path.")


aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)
parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

corners, ids, rejected = detector.detectMarkers(image)

print("Detected markers:", ids)
print(f"Corners: {corners}")
pt1 = (int(corners[0][0][0][0]), int(corners[0][0][0][1]))
pt2 = (int(corners[0][0][3][0]), int(corners[0][0][3][1]))
cv2.line(image,pt1,pt2,(255,0,0),5)

if ids is not None:
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()