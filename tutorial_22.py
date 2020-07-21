import cv2 as cv
import numpy as np


def erode_demo(image):#腐蚀效果
    """
    就像土壤侵蚀一样，这个操作会把前景物体的边界腐蚀掉（但是前景仍然是白色）。
    卷积核沿着图像滑动，如果与卷积核对应的原图像的所有像素值都是 1，
    那么中心元素就保持原来的像素值，否则就变为零。
    这回产生什么影响呢？根据卷积核的大小靠近前景的所有像素都会被腐蚀掉（变为 0），
    所以前景物体会变小，整幅图像的白色区域会减少。
    这对于去除白噪声很有用，也可以用来断开两个连在一块的物体等。
    """
    print(image.shape)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    #cv.imshow("binary demo",binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    dst = cv.erode(binary,kernel=kernel)
    cv.imshow("erode_demo",dst)


def dilate_demo(image):#膨胀效果
    """
    与腐蚀相反，与卷积核对应的原图像的像素值中只要有一个是 1，中心元 素的像素值就是 1。
    所以这个操作会增加图像中的白色区域（前景）。一般在去 噪声时先用腐蚀再用膨胀。
    因为腐蚀在去掉白噪声的同时，也会使前景对象变 小。所以我们再对他进行膨胀。
    这时噪声已经被去除了，不会再回来了，但是 前景还在并会增加。
    膨胀也可以用来连接两个分开的物体。
    """
    print(image.shape)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
   # cv.imshow("binary demo",binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    dst = cv.dilate(binary,kernel=kernel)
    cv.imshow("dilate_demo",dst)

pic = cv.imread("D:/opencv_test_pics/example.jpg")
cv.imshow('origin image',pic)
#erode_demo(pic)
#dilate_demo(pic)
"""
对彩色图片可以直接进行膨胀与腐蚀
kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
dst_erode = cv.erode(pic,kernel)
dst_dilate = cv.dilate(pic,kernel)
cv.imshow("bgr_xxx_demo",xxx)
"""
cv.waitKey(0)
cv.destroyAllWindows()