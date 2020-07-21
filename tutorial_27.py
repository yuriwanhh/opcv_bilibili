import cv2 as cv
import pytesseract as tess
from PIL import Image


"""
验证码识别
1.步骤：
    1. 预处理-去除干扰线和点
    2.不同的结构元素中选择
    3. Image和numpy array相互转换
    4. 识别和输出 tess.image_to_string
"""


def text_detect_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(1,2))
    open1 = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 1))
    open2 = cv.morphologyEx(open1, cv.MORPH_OPEN, kernel)
    #cv.imshow("open2",open2)
    text_image = Image.fromarray(open2)
    print(tess.image_to_string(text_image,lang='chi_sim'))#中文识别要加后缀


pic = cv.imread("D:/opencv_test_pics/word_test2.jpg")
cv.imshow('origin image',pic)

cv.namedWindow('origin image',cv.WINDOW_AUTOSIZE)
#cv.namedWindow('open_demo',cv.WINDOW_AUTOSIZE)
text_detect_demo(pic)

cv.waitKey(0)
cv.destroyAllWindows()