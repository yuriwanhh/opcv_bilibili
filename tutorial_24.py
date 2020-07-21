import cv2 as cv
import numpy as np

"""
其他形态学操作：
顶帽：原图像与开操作之间的差值图像
黑帽：比操作与原图像直接的差值图像
形态学梯度：其实就是一幅图像膨胀与腐蚀的差别。 结果看上去就像前景物体的轮廓
基本梯度：膨胀后图像减去腐蚀后图像得到的差值图像。
内部梯度：用原图减去腐蚀图像得到的差值图像。
外部梯度：膨胀后图像减去原图像得到的差值图像。
"""

def top_hat_demo(image):#顶帽
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(15,15))
    dst = cv.morphologyEx(gray,cv.MORPH_TOPHAT,kernel)
    dst = cv.add(dst,25)
    cv.imshow("top_hat_demo",dst)


def black_hat_demo(image):#黑帽
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(15,15))
    dst = cv.morphologyEx(gray,cv.MORPH_BLACKHAT,kernel)
    dst = cv.add(dst,25)
    cv.imshow("black_hat_demo",dst)


def base_grad_demo(image):#基本梯度
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    dst = cv.morphologyEx(gray,cv.MORPH_GRADIENT,kernel)
    cv.imshow("base_grad_demo",dst)


def inout_grad_demo(image):#基本梯度
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    dm = cv.dilate(image,kernel)#膨胀处理
    em = cv.erode(image,kernel)#腐蚀处理
    dst_in = cv.subtract(image,em)#内部梯度
    dst_out = cv.subtract(dm,image)#外部梯度
    cv.imshow("dst_in_demo",dst_in)
    cv.imshow("dst_out_demo", dst_out)


pic = cv.imread("D:/opencv_test_pics/example.jpg")
cv.imshow('origin image',pic)
inout_grad_demo(pic)
cv.waitKey(0)
cv.destroyAllWindows()