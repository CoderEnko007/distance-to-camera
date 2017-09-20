import numpy as np
import imutils
import argparse
import cv2

def find_marker(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(image, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	cv2.imshow("edged", edged)

	(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	#c = max(cnts, key = cv2.contourArea)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02*peri, True)
		cv2.drawContours(image, [approx], -1, [0, 255, 0], 2)
		
		if len(approx) == 4:
			screenCnt = approx
			break
			
		#cv2.imshow("image", imutils.resize(image, width = 500))
	#cv2.waitKey(0)
	#minAreaRect返回矩形的中心点坐标，长宽，旋转角度度值取值范围为[-90,0)，当矩形水平或竖直时均返回-90
	return cv2.minAreaRect(screenCnt)

def distance_to_camera(knownWidth, focusLength, perWidth):
	return (knownWidth * focusLength) / perWidth

KNOWN_WIDTH = 17.5#24.0
KNOWN_DISTANCE = 43.5#11.0
IMAGE_PATHS = ["images\\test2.jpg", "images\\test1.jpg"]#["images\\2ft.png", "images\\3ft.png", "images\\4ft.png"]

image = cv2.imread(IMAGE_PATHS[0])
marker = find_marker(image)
focusLength = (marker[1][0] * KNOWN_DISTANCE) /KNOWN_WIDTH

for imagePath in IMAGE_PATHS:
	image = cv2.imread(imagePath)
	marker = find_marker(image)
	distance = distance_to_camera(KNOWN_WIDTH, focusLength, marker[1][0])

	box = np.int0(cv2.boxPoints(marker))
	cv2.drawContours(image, [box], -1, [0, 255, 0], 2)
	cv2.putText(image, "%.2ft" % (distance),
		(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)
	cv2.imshow("image", imutils.resize(image, width = 500))
	cv2.waitKey(0)
