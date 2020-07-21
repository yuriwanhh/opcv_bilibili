import cv2 as cv
import numpy as np


def fill_color_demo(image):#彩色图像填充
    coptIMG = image.copy()
    h,w = image.shape[:2]
    mask = np.zeros([h+2,w+2],np.uint8)
    # 参数：原图，mask图，起始点，填充颜色，起始点对应值减去该值作为最低值，起始点值加上该值作为最高值，彩色图模式
    cv.floodFill(coptIMG,mask,(30,30),(0,255,255),(100,100,100),(50,50,50),cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("fill",coptIMG)


def fill_binary_demo():#二值图像填充
    image = np.zeros([400,400,3],np.uint8)
    image[150:250,150:250,:] = 255
    cv.imshow("fill_binary",image)
    mask = np.ones([402,402,1],np.uint8)
    mask[181:221,181:221]=0#只在mask图为零的部分在原图上进行填充，mask中为1部分自动忽略
    cv.floodFill(image,mask,(180,180),(0,0,255),cv.FLOODFILL_MASK_ONLY)
    cv.imshow("fill_binary",image)


pic = cv.imread("D:/le.png")
print(pic.shape)
face = pic[250:400,650:800]
gray = cv.cvtColor(face,cv.COLOR_BGR2GRAY)#转换为灰度图片
backface = cv.cvtColor(gray,cv.COLOR_GRAY2BGR)#灰度转换为bgr时，只是增加了通道数，不能复原颜色
pic[250:400,650:800] = backface
#cv.imshow('new window',pic)
#cv.imshow("face_window",face)
#fill_color_demo(pic)
fill_binary_demo()
cv.waitKey(0)
cv.destroyAllWindows()