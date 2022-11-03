import cv2

image = cv2.imread("./imgs/car2_s.jpg")

rawImage = image.copy()
cv2.imshow("rawImage", rawImage)
cv2.waitKey(0)

Gaussian = cv2.GaussianBlur(image, (3, 3), 0)
# cv2.imshow("Gaussian", Gaussian)
# cv2.waitKey(0)

gray = cv2.cvtColor(Gaussian, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray", gray)
# cv2.waitKey(0)

SobelX = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
absX = cv2.convertScaleAbs(SobelX)
edged = absX
# cv2.imshow("SobelX", edged)
# cv2.waitKey(0)

ret, binary = cv2.threshold(edged, 0, 255, cv2.THRESH_OTSU)
# cv2.imshow("binary", binary)
# cv2.waitKey(0)

kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernelX)
# cv2.imshow("closed", closed)
# cv2.waitKey(0)

kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernelY)
# cv2.imshow("opened", opened)
# cv2.waitKey(0)

median = cv2.medianBlur(opened, 15)
cv2.imshow("median", median)
cv2.waitKey(0)

contours, w = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
cv2.imshow("image", image)
cv2.waitKey(0)

for item in contours:
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    if weight > (height * 3):
        plate = rawImage[y : y + height, x : x + weight]
        cv2.imwrite('./imgs/out2.png', plate)
        cv2.imshow("plate", plate)
        cv2.waitKey(0)

cv2.destroyAllWindows()
