import cv2
# 缩小图片尺寸
def zoom_out(path, out_name):
    img = cv2.imread(path)
    dst = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_name, dst)


