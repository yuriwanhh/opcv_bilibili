import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def hist2d_demo(image):
    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
    #cv.imshow("hist2d",hist)
    plt.imshow(hist,interpolation="nearest")
    plt.title("2d Hist")
    plt.show()

# OpenCV 提供的函数 cv2.calcBackProject() 可以用来做直方图反向投影。
# 它的参数与函数 cv2.calcHist 的参数基本相同。其中的一个参数是我 们要查找目标的直方图。
# 同样再使用目标的直方图做反向投影之前我们应该先对其做归一化处理。
# 返回的结果是一个概率图像
def back_projection_demo():
    sample = cv.imread("D:/opencv_test_pics/sample.jpg")
    target = cv.imread("D:/opencv_test_pics/fruits.jpg")
    sample_hsv = cv.cvtColor(sample,cv.COLOR_BGR2HSV)
    tar_hsv = cv.cvtColor(target,cv.COLOR_BGR2HSV)

    sample_hist = cv.calcHist([sample_hsv],[0,1],None,[32,32],[0,180,0,256])
    #calcHist(images, channels, mask, histSize, ranges[调节精确度, hist[, accumulate]]) -> hist

    cv.normalize(sample_hist,sample_hist,0,255,cv.NORM_MINMAX)#归一化
    # 归一化：原始图像，结果图像，映射到结果图像中的最小值，最大值，归一化类型
    # cv.NORM_MINMAX对数组的所有值进行转化，使它们线性映射到最小值和最大值之间
    # 归一化后的图像便于显示，归一化后到0,255之间了

    dst = cv.calcBackProject([tar_hsv],[0,1],sample_hist,[0,180,0,256],1)

    cv.imshow("sample",sample)
    cv.imshow("target", target)
    cv.imshow("backProject",dst)
    """
    将提取结果转换为3通道图片，与原图按位与运算
    dst_3d = cv.merge((dst, dst, dst))#图片单通道转三通道
    dst_and = cv.bitwise_and(dst_3d, target)  # 与
    cv.imshow("test",dst_and)
    """


pic = cv.imread("D:/opencv_test_pics/example.jpg")
#cv.imshow('origin image',pic)
#hist2d_demo(pic)
back_projection_demo()
cv.waitKey(0)
cv.destroyAllWindows()