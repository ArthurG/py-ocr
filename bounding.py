import cv2
import numpy as np

img = cv2.imread("5.png")

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# invert
thresh = 255 - thresh

# apply horizontal morphology close
kernel = np.ones((5 ,191), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# get external contours
contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
print(contours)

# draw contours
result = gray.copy()
for cntr in contours:
    # get bounding boxes
    pad = 10
    x,y,w,h = cv2.boundingRect(cntr)
    cv2.rectangle(result, (x-pad, y-pad), (x+w+pad, y+h+pad), (255, 0, ), 4)

# save result
cv2.imwrite("6.png",result)
cv2.imwrite("7.png",morph)