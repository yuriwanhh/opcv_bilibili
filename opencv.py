import cv2 as cv
import numpy as np

def video_demo():
    capture = cv.VideoCapture(0)#0代表默认摄像头
    while True:
        ret,frame= capture.read()
        frame = cv.flip(frame,1)
        cv.imshow("video",frame)
        c = cv.waitKey(1)
        if c == 27:
            break


def get_image_info(image):
    print(type(image))
    print(image.shape)
    print(image.size)
    print(image.dtype)
    pixel_data = np.array(image)
    print(pixel_data)


pic = cv.imread("D:/le.png")
cv.namedWindow('1',cv.WINDOW_AUTOSIZE)
cv.imshow('1',pic)
get_image_info(pic)
gray = cv.cvtColor(pic,cv.COLOR_BGR2GRAY)
#cv.imwrite("D:/test.jpg",gray)
video_demo()
cv.waitKey()
cv.destroyAllWindows()


