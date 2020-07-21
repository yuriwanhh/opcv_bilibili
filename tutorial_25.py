import cv2 as cv
import numpy as np

"""
分水岭算法原理：
  任何一副灰度图像都可以被看成拓扑平面，灰度值高的区域可以被看成是山峰，灰度值低的区域可以被看成是山谷。
我们向每一个山谷中灌不同颜色的水。随着水的位的升高，不同山谷的水就会相遇汇合，为了防止不同山谷的水汇合，
我们需要在水汇合的地方构建起堤坝。不停的灌水，不停的构建堤坝直到所有的山峰都被水淹没。
我们构建好的堤坝就是对图像的分割。这就是分水岭算法的背后哲理
  但是这种方法通常都会得到过度分割的结果，这是由噪声或者图像中其他不规律的因素造成的。
为了减少这种影响，OpenCV 采用了基于掩模的分水岭算法，在这种算法中我们要设置哪些山谷点会汇合，哪些不会。
这是一种交互式的图像分割。我们要做的就是给我们已知的对象打上不同的标签。
如果某个区域肯定是前景或对象，就使用某个颜色（或灰度值）标签标记它。
如果某个区域肯定不是对象而是背景就使用另外一个颜色标签标记。而剩下的不能确定是前景还是背景的区域就用 0 标记。
这就是我们的标签。然后实施分水岭算法。
每一次灌水，我们的标签就会被更新，当两个不同颜色的标签相遇时就构建堤坝，直到将所有山峰淹没，
最后我们得到的边界对象（堤坝）的值为 -1。
基于距离的分水岭分割流程：
输入图像->灰度->二值->距离变换->寻找种子->生成marker->分水岭变换->输出图像
"""

def watershed_demo(image):

    #blur/gray/binary image
    blurred = cv.pyrMeanShiftFiltering(image,10,100)
    gray = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    #morphology operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)  # 连续两次开操作（去除图像中的任何小的白噪声）；闭运算（为了去除物体上的小洞）
    sure_bg = cv.dilate(opening, kernel, iterations=3)  # 连续三次膨胀操作,确定背景区域


    # Finding sure foreground area
    # 距离变换的基本含义是计算一个图像中非零像素点到最近的零像素点的距离，
    # 也就是到零像素点的最短距离
    # 最常见的距离变换算法就是通过连续的腐蚀操作来实现，腐蚀操作的停止条件是所有前景像素都被完全腐蚀。
    # 这样根据腐蚀的先后顺序，我们就得到各个前景像素点到前景中心像素点的距离。
    # 根据各个像素点的距离值，设置为不同的灰度值。这样就完成了二值图像的距离变换
    # cv2.distanceTransform(src, distanceType, maskSize)
    # 第二个参数 0,1,2 分别表示 CV_DIST_L1, CV_DIST_L2 , CV_DIST_C

    # distance transform  确定前景区域
    dist = cv.distanceTransform(opening,cv.DIST_L2,3)
    #dist_out = cv.normalize(dist,0,1.0,cv.NORM_MINMAX)#归一化处理
    #cv.imshow("dist_out demo",dist_out*50)
    ret,surface = cv.threshold(dist,dist.max()*0.7,255,cv.THRESH_BINARY)
    surface_fg = np.uint8(surface)

    #查找未知区域
    unknown = cv.subtract(sure_bg,surface_fg)

    #watershed transform 标记标签
    ret,markers = cv.connectedComponents(surface_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(image,markers=markers)
    image[markers == -1]=[0,0,255]

    cv.imshow("result",image)


pic = cv.imread("D:/opencv_test_pics/circle.jpg")
cv.imshow('origin image',pic)
watershed_demo(pic)
cv.waitKey(0)
cv.destroyAllWindows()