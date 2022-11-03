import cv2

image = cv2.imread("./imgs/out.png")

Gaussian = cv2.GaussianBlur(image, (3, 3), 0)
cv2.imshow("Gaussian", Gaussian)
cv2.waitKey()

gray = cv2.cvtColor(Gaussian, cv2.COLOR_RGB2GRAY)
cv2.imshow("gray", gray)
cv2.waitKey()

ret, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
cv2.imshow("bin", bin)
cv2.waitKey()


# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# erode = cv2.erode(bin, kernel)
# cv2.imshow("erode", erode)
# cv2.waitKey()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilate = cv2.dilate(bin, kernel)
cv2.imshow("dilate", dilate)
cv2.waitKey()


contours, hierarchy = cv2.findContours(
    dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
# cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
# cv2.imshow("image", image)
# cv2.waitKey()

chars = []
for item in contours:
    rect = cv2.boundingRect(item)
    x, y, w, h = cv2.boundingRect(item)
    chars.append(rect)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)

chars = sorted(chars, key=lambda s: s[0], reverse=False)
plateChars = []
for i, word in enumerate(chars):
    x, y, w, h = word
    plateChar = image[y : y + h, x : x + w]
    cv2.imshow("image" + str(i), plateChar)
    cv2.waitKey()
    if (h > (w * 1.5)) and (h < (w * 8)) and (w > 3):
        plateChar = image[y : y + h, x : x + w]
        plateChars.append(plateChar)

# for i, im in enumerate(plateChars):
#     cv2.imshow("char" + str(i), im)
#     cv2.waitKey()

cv2.destroyAllWindows()
