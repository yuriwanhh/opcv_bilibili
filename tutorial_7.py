import cv2 as cv
import numpy as np


def clamp(pv):#保证不超过255范围
    if pv>255:
        return 255
    if pv<0:
        return 0
    else:
        return pv


def gaussian_noise(image):#增加高斯噪点
    h,w,c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0,20,3)#随机噪点
            b = image[row,col,0]
            g = image[row,col,1]
            r = image[row,col,2]
            #为图片添加随机噪点
            image[row, col, 0] =clamp(b+s[0])
            image[row, col, 1] =clamp(g+s[1])
            image[row, col, 2] =clamp(r+s[2])
    cv.imshow("noise image",image)
pic = cv.imread("D:/le.png")
cv.imshow('origin image',pic)
t1 = cv.getCPUTickCount()
gaussian_noise(pic)
t2 = cv.getCPUTickCount()
time = (t2-t1)/cv.getTickFrequency()
print("time : %f ms"%(time*1000))
#高斯模糊后两个参数有一个为零，用一个参数可求出另一个
dst = cv.GaussianBlur(pic,(5,5),0)
cv.imshow("gaussian Blur",dst)
cv.waitKey(0)
cv.destroyAllWindows()