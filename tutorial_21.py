import cv2 as cv
import numpy as np


def measure_demo(image):
    dst = cv.GaussianBlur(image,(3,3),0)
    gray = cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
    cv.imshow("binary_demo",binary)
    DST = cv.cvtColor(binary,cv.COLOR_GRAY2BGR)
    contours,hireachy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        area = cv.contourArea(contour)#计算轮廓面积
        print("contour area: %s" % area)
        x,y,w,h = cv.boundingRect(contour)
        rate = min(w, h) / max(w, h)#所得矩形的纵横比
        print("rectangel rate : %s"%rate)
        mm = cv.moments(contour)
        if mm['m00'] == 0:#除数为零跳过
            continue
        else:
            # 计算出对象的重心
            cx = mm['m10']/mm['m00']
            cy = mm['m01']/mm['m00']
            cv.circle(image,(np.int(cx),np.int(cy)),2,(0,255,255),-1)#用实心圆画出重心
            cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)#画出轮廓矩形框

            """cv.approxPolyDP(contour, epsilon, True) 参数解释
             .   @param curve Input vector of a 2D point stored in std::vector or Mat
             .   @param approxCurve Result of the approximation. The type should match the type of the input curve.
             .   @param epsilon Parameter specifying the approximation accuracy. This is the maximum distance
             .   between the original curve and its approximation.
             .   @param closed If true, the approximated curve is closed (its first and last vertices are
             .   connected). Otherwise, it is not closed.
                    """
            # 将轮廓形状近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来决定。
            # 为了帮助理解，假设从一幅图像中查找一个矩形，但是由于图像的种种原因，我们不能得到一个完美的矩形，
            # 而是一个“坏形状”。
            # 现在你就可以使用这个函数来近似这个形状了。
            # 这个函数的第二个参数叫 epsilon，它是从原始轮廓到近似轮廓的最大距离。
            # 它是一个准确度参数。选择一个好的 epsilon 对于得到满意结果非常重要。

            approxCure = cv.approxPolyDP(contour,3,True)
            #以下if判断为所绘制逼近图形框需要多少条逼近直线
            if approxCure.shape[0]>10:#逼近线段较多，一般为弧线轮廓
                cv.drawContours(DST,contours,i,(0,0,255),3)
            if approxCure.shape[0] == 4:#矩形用四条线逼近
                cv.drawContours(DST, contours, i, (0, 255, 0), 3)
            if approxCure.shape[0]==3:#三角形用三条直线逼近
                cv.drawContours(DST,contours,i,(255,0,0),3)
            cv.imshow("DST_demo",DST)

    cv.imshow("measure_demo",image)


pic = cv.imread("D:/opencv_test_pics/pic1.png")
cv.imshow('origin image',pic)
measure_demo(pic)
cv.waitKey(0)
cv.destroyAllWindows()