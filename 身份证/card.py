import cv2
import numpy as np

card = cv2.imread("./imgs/id_card_b_s.jpg")
gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gaussian, 50, 200)
cts, hierarchy = cv2.findContours(
    edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

cts = sorted(cts, key=cv2.contourArea, reverse=True)
ct = cts[0]
peri = 0.01 * cv2.arcLength(ct, True)
approx = cv2.approxPolyDP(ct, peri, True)
# print(approx.reshape(4, 2))
pts = approx.reshape(4, 2)
xSorted = pts[np.argsort(pts[:, 0]), :]
left = xSorted[:2, :]
right = xSorted[2:, :]
left = left[np.argsort(left[:, 1]), :]
(tl, bl) = left
right = right[np.argsort(right[:, 1]), :]
(tr, br) = right
src = np.array([tl, tr, br, bl], dtype=np.float32)
widthA = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
widthB = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
maxWidth = max(int(widthA), int(widthB))
heightA = np.sqrt((br[0] - tr[0]) ** 2 + (br[1] - tr[1]) ** 2)
heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
maxHeight = max(int(heightA), int(heightB))
# srcx = src.reshape((-1,1,2))
print(maxWidth)
print(maxHeight)
dst = np.array(
    [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
    dtype=np.float32,
)
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(card, M, (maxWidth, maxHeight))

# cv2.polylines(card, [src], True, (255, 0, 0), 8)
cv2.imshow("result", warped)


cv2.waitKey()
cv2.destroyAllWindows()
