import cv2 as cv
import numpy as np


def extract_object_demo():
    capture = cv.VideoCapture(0)
    while True:
        ret,frame = capture.read()
        frame = cv.flip(frame, 1)
        if ret == False:
            break
        hsv_lower = np.array([0,0,0])#对应颜色最小值
        hsv_higher = np.array([180,255,46])#对应颜色最大值
        mask = cv.inRange(frame,lowerb=hsv_lower,upperb=hsv_higher)
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        dst = cv.bitwise_and(frame,frame,mask=mask)#将颜色单独提取出来
        #cv.imshow("video",frame)
        cv.imshow("mask", mask)
        cv.imshow("dst", dst)


        c = cv.waitKey(40)
        if c == 27:
            break



def color_space_demo(image):#色彩空间转换
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)#BGR转换为灰度
    cv.imshow("gray",gray)
    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)#BGR转换为HSV H:0-180 S/V:0-255
    cv.imshow("hsv",hsv)
    yuv = cv.cvtColor(image,cv.COLOR_BGR2YUV)#BGR转换为YUV
    cv.imshow("yuv",yuv)
    Ycrcb = cv.cvtColor(image,cv.COLOR_BGR2YCrCb)#BGR转化为Ycrcb
    cv.imshow("Ycrcb",Ycrcb)


pic = cv.imread("D:/le.png")
#cv.imshow("BGR",pic)#展示图片
#color_space_demo(pic)#色彩空间转换
"""#通道拆分与合并
b,g,r = cv.split(pic)
cv.imshow("blue",b)
cv.imshow("green",g)
cv.imshow("red",r)
pic[:,:,2]=0   #2通道赋值为0
cv.imshow("change",pic)
sum_pic = cv.merge([b,g,r])#通道合并
cv.imshow("sum",sum_pic)
"""
extract_object_demo()#视频颜色提取

cv.waitKey()
cv.destroyAllWindows()
