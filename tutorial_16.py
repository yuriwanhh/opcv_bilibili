import cv2 as cv
import numpy as np


# 图像梯度（由x,y方向上的偏导数和偏移构成），有一阶导数（sobel算子）和二阶导数（Laplace算子）
# 用于求解图像边缘，一阶的极大值，二阶的零点
# 一阶偏导在图像中为一阶差分，再变成算子（即权值）与图像像素值乘积相加，二阶同理


def lapalain_demo(image):
    dst = cv.Laplacian(image,cv.CV_32F)#卷积核为4领域
    """
    kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])#手动定义拉普拉斯卷积核(所用为8领域卷积核，还可设置为4领域)
    dst = cv.filter2D(image,cv.CV_32F,kernel=kernel)
    """
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("lapalain_demo",lpls)


def sobel_demo(image):
    grad_x = cv.Sobel(image,cv.CV_32F,1,0)
    grad_y = cv.Sobel(image,cv.CV_32F,0,1)
    #grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)
    #grad_y = cv.Scharr(image, cv.CV_32F, 0, 1) Scharr算子为Sobel算子的增强版,增强边缘
    gradx = cv.convertScaleAbs(grad_x)# 由于算完的图像有正有负，所以对其取绝对值
    grady = cv.convertScaleAbs(grad_y)
    #cv.imshow("grad_x",gradx)
    #cv.imshow("grad_y",grady)

    # 计算两个图像的权值和，dst = src1*alpha + src2*beta + gamma
    gradxy = cv.addWeighted(gradx,0.5,grady,0.5,0)
    cv.imshow("grad_all",gradxy)
pic = cv.imread("D:/opencv_test_pics/lena.jpg")
cv.imshow('origin image',pic)
sobel_demo(pic)
cv.waitKey(0)
cv.destroyAllWindows()