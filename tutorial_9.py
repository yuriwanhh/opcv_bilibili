import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def plot_demo(image):
    plt.hist(image.ravel(),256,[0,256])
    plt.show()


def image_hist(image):
    color = ('blue','green','red')
    for i,color in enumerate(color):
        hist = cv.calcHist([image], [i] , None ,[256],[0,256])#https://blog.csdn.net/YZXnuaa/article/details/79231817
        plt.plot(hist,color = color)
        plt.xlim([0,256])
    plt.show()


pic = cv.imread("D:/opencv_test_pics/example.jpg")
cv.imshow('origin image',pic)
image_hist(pic)
cv.waitKey(0)
cv.destroyAllWindows()