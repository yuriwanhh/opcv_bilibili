import cv2 as cv
import numpy as np

#处理的图像必须为正方形，边长为2^n
# 图像金字塔和拉普拉斯金字塔(L1 = g1 - expand(g2))：reduce：高斯模糊+降采样，expand：扩大+卷积
# PyrDown降采样，PyrUp还原
def pyramid_demo(image):
    level = 3
    temp = image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("reduce_demo"+str(i),dst)
        temp = dst.copy()
    return pyramid_images


def lapalian_demo(image):
    pyramid_images = pyramid_demo(image)
    level = len(pyramid_images)
    for i in range(level):
        if (i-1) < 0:
            expand = cv.pyrUp(pyramid_images[i],dstsize=image.shape[:2])
            lpls = cv.subtract(image,expand)
            cv.imshow("lapalian_down_"+str(i),lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i - 1].shape[:2])
            lpls = cv.subtract(pyramid_images[i - 1], expand)
            cv.imshow("lapalian_down_" + str(i), lpls)


pic = cv.imread("D:/opencv_test_pics/lena.jpg")
cv.imshow('origin image',pic)
lapalian_demo(pic)
cv.waitKey(0)
cv.destroyAllWindows()