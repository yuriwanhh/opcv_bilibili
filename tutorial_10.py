import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def equalhist_demo(image):#全局直方图均衡化,自动调整对比度，增强图像
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)#彩色图转换为灰度图，或指定彩色图某个通道
    dst = cv.equalizeHist(gray)
    cv.imshow("equal hist",dst)

def clahe_demo(image):#局部增强对比度
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 彩色图转换为灰度图，或指定彩色图某个通道
    clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    dst = clahe.apply(gray)
    cv.imshow("equal hist", dst)


# 自己创建直方图，相当于建了一个API
def create_rgb_demo(image):
    h, w, c = image.shape
    rgbHist = np.zeros([16*16*16, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b/bsize)*16*16 + np.int(g/bsize)*16 + np.int(r/bsize)
            rgbHist[np.int(index), 0] = rgbHist[np.int(index), 0] + 1

    return rgbHist


def hist_compare(image1,image2):#图像大小一致时的比较方法，不一致时要进行归一化
    hist1 = create_rgb_demo(image1)
    hist2 = create_rgb_demo(image2)
    match1 = cv.compareHist(hist1,hist2,cv.HISTCMP_BHATTACHARYYA)#巴氏距离
    match2 = cv.compareHist(hist1,hist2,cv.HISTCMP_CORREL)#相关性
    match3 = cv.compareHist(hist1,hist2,cv.HISTCMP_CHISQR)#卡方
    print("巴氏距离：%s，相关性：%s，卡方：%s"%(match1,match2,match3))



pic = cv.imread("D:/opencv_test_pics/example.jpg")
#cv.imshow('origin image',pic)
#clahe_demo(pic)
image1 = cv.imread("D:/opencv_test_pics/lena_tmpl.jpg")
image2 = cv.imread("D:/opencv_test_pics/lena.jpg")
hist_compare(image1,image2)
cv.waitKey(0)
cv.destroyAllWindows()