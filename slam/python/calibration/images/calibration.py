# 这个是opencv官网上面给的标定代码
import numpy as np
from collections import Counter
import cv2 as cv
import glob

print(cv.TERM_CRITERIA_EPS)
print(cv.TERM_CRITERIA_MAX_ITER)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

height, width = 11, 8
objp = np.zeros((height * width, 3), np.float32)
objp[:, :2] = np.mgrid[0:height, 0:width].T.reshape(-1, 2)

objp = objp * 0.03

objpoints = []
imgpoints = []

images = glob.glob('*.jpg')
print('图片的数量是', len(images))

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (height, width), None)
    if ret == True:
        objpoints.append(objp)
        # 经测试 cornerSubPix之后，corners的值并没有发生改变，为什么objpoints和imgpoints里面的内容要一摸一样呢
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        cv.drawChessboardCorners(img, (height, width), corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey(0)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(ret)
print(mtx)

#for fname in images:
#    img = cv.imread(fname)
#    dst = cv.undistort(img, mtx, dist, None, mtx)
#    h, w = img.shape[:2]
#    new_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
#    dst2 = cv.undistort(img, mtx, dist, None, new_mtx)
#    x, y, w, h = roi
#    dst2 = dst2[y:y+h, x:x+w]
#    cv.imshow("img", img)
#    cv.imshow("dst", dst)
#    cv.imshow("dst2", dst2)
#    cv.waitKey(0)
#    #cv.imwrite('undistort/' + fname, dst2)
#
#print(new_mtx)
