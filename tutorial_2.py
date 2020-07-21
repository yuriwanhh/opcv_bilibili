import cv2 as cv
import numpy as np


def access_pixels(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    print("height : %s,width : %s,channels ; %s"%(height,width,channels))
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row,col,c]
                image[row,col,c] = 255 - pv#像素取反
    cv.imshow("pic_test",image)


def inverse(image):
    dst = cv.bitwise_not(image)
    cv.imshow("inverse image",dst)


def creat_image():
    """""
    img = np.zeros([400,400,3],np.uint8)#创建一个400*400三通道8位图像
    img[: ,:,0]=np.ones([400,400])*144#给第一通道各个像素点赋值，通道顺序是B G R
    cv.imshow("new image",img)
    """""
    img = np.ones([200,200,1],np.uint8)#单通道图像创建
    img = img*10
    cv.imshow("new image", img)
pic = cv.imread("D:/le.png")
#cv.namedWindow('1',cv.WINDOW_AUTOSIZE)
#cv.imshow('1',pic)

t1 = cv.getCPUTickCount()
inverse(pic)
#access_pixels(pic)

#creat_image()


"""
#用np创建数组
m1 = np.ones([3,3],np.float32)
m1.fill(111.1)
print(m1)
m2 = m1.reshape([1,9])#转换维度，不损失数值
print(m2)
m3 = np.array([[2,3,4],[5,6,7],[1,8,9]],np.int32)#用np创建数组
m3.fill(9)
print(m3)
"""
t2 = cv.getCPUTickCount()
print("time : %f ms"%(((t2-t1)/cv.getTickFrequency())*1000))
cv.waitKey()
cv.destroyAllWindows()