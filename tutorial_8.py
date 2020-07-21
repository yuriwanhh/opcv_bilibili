import cv2 as cv
import numpy as np


def bi_demo(image):#高斯双边模糊
    """
       同时考虑空间与信息和灰度相似性，达到保边去噪的目的
       双边滤波的核函数是空间域核与像素范围域核的综合结果：
       在图像的平坦区域，像素值变化很小，对应的像素范围域权重接近于1，此时空间域权重起主要作用，相当于进行高斯模糊；
       在图像的边缘区域，像素值变化很大，像素范围域权重变大，从而保持了边缘的信息。
    """
    """
    #第三个参数为颜色范围，这个参数的值越大，
    表明该像素邻域内有月宽广的颜色会被混合到一起，产生较大的半相等颜色区域。
    第四个为空间范围，如果该值较大，则意味着颜色相近的较远的像素将相互影响，
    从而使更大的区域中足够相似的颜色获取相同的颜色。
    """
    dst = cv.bilateralFilter(image,0,100,15)
    cv.imshow("bi_dome",dst)


def shift_demo(image):#均值迁移模糊
    dst = cv.pyrMeanShiftFiltering(image,10,10)
    cv.imshow("shift image",dst)


pic = cv.imread("D:/opencv_test_pics/example.jpg")
cv.imshow('origin image',pic)
cv.namedWindow('origin image',cv.WINDOW_AUTOSIZE)
bi_demo(pic)
shift_demo(pic)
cv.waitKey(0)
cv.destroyAllWindows()