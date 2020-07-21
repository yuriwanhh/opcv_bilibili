import cv2 as cv
import numpy as np


def bid_image_threshold(image):#超大图像二值化
    ch = 256#步长
    cw = 256#步长
    h,w = image.shape[:2]
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    for row in range(0,h,ch):
        for col in range(0,w,cw):
            roi = gray[row:row+ch,col:col+cw]# 将一张图片每隔ch * cw分成一份
            dst=cv.adaptiveThreshold(roi,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,127,20)#局部二值化
            gray[row:row + ch, col:col + cw] = dst#将处理后的图像值赋给原来大图
            print(np,str(dst),np.mean(dst))#每个小图像的方差以及均值，可以根据方差对图像进行过滤
            """过滤方法
            if np.str(roi) < x：
                gray[row:row + ch, col:col + cw] = 255
            else:
                dst=cv.adaptiveThreshold(roi,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,127,20)#局部二值化
                gray[row:row + ch, col:col + cw] = dst
            """
    #cv.imwrite("D:/opencv_test_pics/large_pic_threshold.png",gray)将图像保存显示
pic = cv.imread("D:/opencv_test_pics/example.jpg")
cv.imshow('origin image',pic)
bid_image_threshold(pic)
cv.waitKey(0)
cv.destroyAllWindows()