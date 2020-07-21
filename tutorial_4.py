import cv2 as cv
import numpy as np


def add_demo(m1,m2):
    dst = cv.add(m1,m2)
    cv.imshow("add_demo",dst)


def subtract_demo(m1,m2):
    dst = cv.subtract(m1,m2)
    cv.imshow("subtract_demo",dst)


def multiply_demo(m1,m2):
    dst = cv.multiply(m1,m2)
    cv.imshow("multiply_demo",dst)


def divide_demo(m1,m2):
    dst = cv.divide(m1,m2)
    cv.imshow("divide_demo",dst)


def logic_demo(m1,m2):
    #dst = cv.bitwise_and(m1,m2)#与
    #dst = cv.bitwise_or(m1,m2)#或
    #cv.imshow("logic_demo",dst)#按位取反
    png = cv.imread("D:/le.png")
    dst = cv.bitwise_not(png)
    cv.imshow("bit_not",dst)


def contrast_brightness_demo(image,c,b):
    h, w, ch = image.shape
    blank = np.zeros([h,w,ch],image.dtype)#尺寸相同全黑图片
    dst = cv.addWeighted(image,c,blank,1-c,b)#参数：第一张图，权重，第二张图，权重，亮度
    cv.imshow("contrast_brightness_demo",dst)



def others(m1,m2):#求均值与方差
    M1,dev1 = cv.meanStdDev(m1)
    M2,dev2 = cv.meanStdDev(m2)
    print(M1)
    print(M2)
    print(dev1)
    print(dev2)


src1 = cv.imread("D:/opencv_test_pics/WindowsLogo.jpg")
src2 = cv.imread("D:/opencv_test_pics/LinuxLogo.jpg")
#print(src1.shape)
#print(src2.shape)
cv.namedWindow("image1",cv.WINDOW_AUTOSIZE)
cv.imshow("image1",src1)
cv.imshow("image2",src2)
#add_demo(src1,src2)
#subtract_demo(src2,src1)
#multiply_demo(src1,src2)
#divide_demo(src1,src2)
#others(src1,src2)
#logic_demo(src1,src2)
contrast_brightness_demo(src1,1.2,10)#对比度为1.2，亮度每个像素加10
cv.waitKey()
cv.destroyAllWindows()