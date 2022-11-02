import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

img = cv2.imread("./imgs/finger_5_s.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_skin = np.array([0, 28, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower_skin, upper_skin)
plt.subplot(2, 2, 1)
plt.imshow(mask)

# kernel = np.ones((3, 3), np.uint8)
# mask = cv2.dilate(mask, kernel, iterations=4)
# mask = cv2.GaussianBlur(mask, (5, 5), 100)

kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.dilate(mask, kernel, iterations=2)
plt.subplot(2, 2, 2)
plt.imshow(mask)
plt.show()

contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=lambda x: cv2.contourArea(x))
areacnt = cv2.contourArea(cnt)
hull = cv2.convexHull(cnt)
areahull = cv2.contourArea(hull)
arearatio = areacnt / areahull
print("arearatio=", arearatio)
hull = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull)
n = 0
for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    # 1弧度为57度
    angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
    if angle <= 90 and d > 20:
        n += 1
        cv2.circle(img, far, 3, [255, 0, 0], -1)
    cv2.line(img, start, end, [0, 255, 0], 2)
result = None
if n == 0:
    if arearatio > 0.9:
        result = "0"
    else:
        result = "1"
elif n == 1:
    result = "2"
elif n == 2:
    result = "3"
elif n == 3:
    result = "4"
elif n == 4:
    result = "5"
org = (0, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 0, 255)
thickness = 3

cv2.putText(img, result, org, font, fontScale, color, thickness)
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()
