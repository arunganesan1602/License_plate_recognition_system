import cv2 as cv
import imutils
import numpy as np
import pytesseract as tess
from datetime import datetime
import csv

# To capture the image
cam = cv.VideoCapture(0)
cv.namedWindow("input")
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv.imshow("input", frame)

    k = cv.waitKey(1)

    # To set the function
    if k%256 == 27:
        # ESC pressed
        print("Camera_Closing")
        break

    elif k%256 == 32:
        # SPACE pressed
        img_name = "frame_{}.png".format(img_counter)
        cv.imwrite(img_name, frame)
        print("Uploaded")
        img_counter += 1

cam.release()

# read the image
image = cv.imread("frame_0.png")

# Resize the image to the required size and grayscale it
image = cv.resize(image, (620,480))
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
cv.imshow('grayscale_out',gray)

# Filter the noise in image
gray = cv.bilateralFilter(gray,11,17,17)
cv.imshow('bilateralFilter_out',gray)

# Threshold to get pixel in black pixel
ret, thresh = cv.threshold(gray,127,255,0)
cv.imshow('thresh_out',thresh)

# To find the edges in shape
cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key= cv.contourArea, reverse= True)
screenCut = None

# To get the rectangular shape
for c in cnts:
    peri = cv.arcLength(c,True)
    approx = cv.approxPolyDP(c, 0.018*peri, True)
    if len(approx) == 4:
        screenCut =approx
        break

# masking is used to modify the image
mask = np.zeros(gray.shape,np.uint8)
new_image = cv.drawContours(mask,[screenCut],0 , 255, -1)
new_image = cv.bitwise_and(image, image, mask=mask)
cv.imshow('new_image_out',new_image)

# It is used to copped the image
(x,y) = np.where(mask ==255)
(topx,topy) = (np.min(x),np.min(y))
(bottomx,bottomy) = (np.max(x),np.max(y))
cropped = gray[topx:bottomx+1, topy:bottomy+1]
cv.imshow('cropped_out',cropped)

#It is used to print the license plate number in python console
text = tess.image_to_string(cropped)
Detected_Number = "".join(text.split())
print("Detected Number is :", Detected_Number)

# To save the license plate number in CSV file
file = open("data.csv","a",newline="")
now = datetime.now()
today = datetime.today()
dateStr = today.strftime('%d/%m/%y')
timeStr = now.strftime('%H:%M:%S')
data = (dateStr, timeStr, Detected_Number)
writer = csv.writer(file)
writer.writerow(data)
file.close()

# Waitkey for displaying image
cv.waitKey(10000)
cv.destroyAllWindows()