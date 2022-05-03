import cv2
import sys

if len(sys.argv) != 2:
    raise IOError("输入参数个数应为1")

path = sys.argv[1]

img = cv2.imread(path)
img[0:100, 0:100] = 255

print(img.dtype)

cv2.imshow("image", img)
cv2.waitKey(0)

