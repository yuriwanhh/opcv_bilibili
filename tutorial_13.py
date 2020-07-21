import cv2 as cv
import numpy as np

# 二值图像就是将灰度图转化成黑白图，没有灰，在一个值之前为黑，之后为白
# 有全局和局部两种
# 在使用全局阈值时，我们就是随便给了一个数来做阈值，那我们怎么知道我们选取的这个数的好坏呢？答案就是不停的尝试。
# 如果是一副双峰图像（简 单来说双峰图像是指图像直方图中存在两个峰）呢？
# 我们岂不是应该在两个峰之间的峰谷选一个值作为阈值？这就是 Otsu 二值化要做的。
# 简单来说就是对 一副双峰图像自动根据其直方图计算出一个阈值。
# （对于非双峰图像，这种方法 得到的结果可能会不理想）。

def threshold_demo(image):#全局阈值
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    # 这个函数的第一个参数就是原图像，原图像应该是灰度图。
    # 第二个参数就是用来对像素值进行分类的阈值。可手动设置阈值
    # 第三个参数就是当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
    # 第四个参数来决定阈值方法，见threshold_simple()  |之后cv.THRESH_OTSU为自动寻找阈值方法
    # ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)#此方式为手动设置阈值

    print("threshold value: %s"%ret)
    cv.imshow("threshold demo",binary)


def local_threshold(image):#局部二值化
    # 在前面的部分我们使用是全局阈值，整幅图像采用同一个数作为阈值。
    # 当时这种方法并不适应与所有情况，尤其是当同一幅图像上的不同部分的具有不同亮度时。
    # 这种情况下我们需要采用自适应阈值。此时的阈值是根据图像上的 每一个小区域计算与其对应的阈值。
    # 因此在同一幅图像上的不同区域采用的是不同的阈值，从而使我们能在亮度不同的情况下得到更好的结果。

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #Block Size - 邻域大小（用来计算阈值的区域大小）。
    binary = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,25,10)
    # 这种方法需要我们指定三个参数，返回值只有一个
    # _MEAN_C：阈值取自相邻区域的平均值,_GAUSSIAN_C：阈值取值相邻区域 的加权和，权重为一个高斯窗口。
    # Block Size - 邻域大小（用来计算阈值的区域大小）。一定为奇数。
    # C - 这就是是一个常数，阈值就等于的平均值或者加权平均值减去这个常数。
    cv.imshow("threshold demo", binary)


def custom_threshold(image):#手动计算阈值方法
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h,w = gray.shape[:2]
    m = np.reshape(gray,[1,w*h])
    mean = m.sum()/(w*h)
    ret, binary = cv.threshold(gray, mean , 255, cv.THRESH_BINARY)
    cv.imshow("threshold demo", binary)

pic = cv.imread("D:/opencv_test_pics/example.jpg")
cv.imshow('origin image',pic)
custom_threshold(pic)
cv.waitKey(0)
cv.destroyAllWindows()