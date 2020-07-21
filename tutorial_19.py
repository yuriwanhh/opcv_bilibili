import cv2 as cv
import numpy as np

# Hough Circle 在xy坐标系中一点对应Hough坐标系中的一个圆，xy坐标系中圆上各个点对应Hough坐标系各个圆，
# 相加的一点，即对应xy坐标系中圆心
# 现实考量：Hough圆对噪声比较敏感，所以做hough圆之前要中值滤波，
# 基于效率考虑，OpenCV中实现的霍夫变换圆检测是基于图像梯度的实现，分为两步：
# 1. 检测边缘，发现可能的圆心候选圆心开始计算最佳半径大小
# 2. 基于第一步的基础上，从


def circle_detection(image):
    dst = cv.pyrMeanShiftFiltering(image,10,100)#均值漂移滤波
    gray = cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
    """
    #HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None)
    dp : 累加器图像的分辨率。这个参数允许创建一个比输入图像分辨率低的累加器。
    （这样做是因为有理由认为图像中存在的圆会自然降低到与图像宽高相同数量的范畴）。
    如果dp设置为1，则分辨率是相同的；如果设置为更大的值（比如2），累加器的分辨率受此影响会变小（此情况下为一半）。
    dp的值不能比1小。
    minDist : 该参数是让算法能明显区分的两个不同圆之间的最小距离。**灵活调整
    param1 : 用于Canny的边缘阀值上限，下限被置为上限的一半。
    param2 : 累加器的阈值
    
    """
    circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,150,param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv.circle(image,(i[0],i[1]),i[2],(0,0,255),2)
        cv.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
    cv.imshow("circles",image)

pic = cv.imread("D:/opencv_test_pics/circle.jpg")
cv.imshow('origin image',pic)
circle_detection(pic)
cv.waitKey(0)
cv.destroyAllWindows()