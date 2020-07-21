import cv2 as cv
import numpy as np


def line_detection(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)  # 很小程度高斯模糊降噪，canny对噪声比较敏感
    if not len(blurred.shape) == 2:
        gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)  # 转换为灰度图片
    else:
        gray = blurred
    edge = cv.Canny(gray, 50, 150,apertureSize=3)

    # cv2.HoughLines()返回值就是（ρ,θ）。ρ 的单位是像素，θ 的单位是弧度。
    # 这个函数的第一个参数是一个二值化图像，所以在进行霍夫变换之前要首先进行二值化，或者进行 Canny 边缘检测。
    # 第二和第三个值分别代表 ρ 和 θ 的精确度。第四个参数是阈值，只有累加其中的值高于阈值时才被认为是一条直线
    # 末尾参数150为检测阈值，太高了容易检测不出来
    # 也可以把它看成能 检测到的直线的最短长度（以像素点为单位）。
    lines = cv.HoughLines(edge,1,np.pi/180,150)
    for line in lines:
        r,th = line[0]
        a = np.cos(th)
        b = np.sin(th)
        x0 = a * r
        y0 = b * r
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*(a))
        cv.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    cv.imshow("image_lines",image)


def line_detect_possilbe(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)  # 很小程度高斯模糊降噪，canny对噪声比较敏感
    if not len(blurred.shape) == 2:
        gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)  # 转换为灰度图片
    else:
        gray = blurred
    edge = cv.Canny(gray, 50, 150, apertureSize=3)
    #末尾两个参数为：检测线段的最先长度，容忍线段间的最大间隙
    lines = cv.HoughLinesP(edge, 1, np.pi/180, 80,minLineLength=50,maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("image_lines", image)


pic = cv.imread("D:/opencv_test_pics/pic6.png")
cv.imshow('origin image',pic)
line_detect_possilbe(pic)
cv.waitKey(0)
cv.destroyAllWindows()