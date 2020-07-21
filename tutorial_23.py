import cv2 as cv
import numpy as np
"""
开运算:先进性腐蚀再进行膨胀就叫做开运算,它被用来去除噪声。
闭运算:先膨胀再腐蚀。它经常被用来填充前景物体中的小洞，或者前景物体上的小黑点。
这里我们用到的函数是 cv2.morphologyEx()。
开闭操作作用：
1. 去除小的干扰块-开操作
2. 填充闭合区间-闭操作
3. 水平或垂直线提取,调整kernel的row，col值差异。
比如：采用开操作，kernel为(1, 15),提取垂直线，kernel为(15, 1),提取水平线，
"""

def open_demo(image):#先腐蚀后膨胀，尽量保持其他元素不变，删除小像素白色干扰块。
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    #cv.getStructuringElement  灵活调整结构元素形状以及结构元素大小，以获得想要的结果
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))#矩形核
    #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)) 圆形/椭圆形核
    dst = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    cv.imshow("open_demo",dst)


def close_demo(image):#尽量保持其他元素不变，填充黑色缺陷，先膨胀后腐蚀
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(binary,cv.MORPH_CLOSE,kernel)
    cv.imshow("close_demo",dst)


pic = cv.imread("D:/opencv_test_pics/example.jpg")
cv.imshow('origin image',pic)
open_demo(pic)
cv.waitKey(0)
cv.destroyAllWindows()