import cv2 as cv

def captureImage():
    cap = cv.VideoCapture(0)
    image = cap.read()[1]
    cv.imshow('VideoCapture',image)

    cv.waitKey()
    cap.release() #释放视屏捕捉设备资源
    cv.destroyAllWindows() #关闭cv 打开的隐藏窗口

def captureVideo():
    cap = cv.VideoCapture(0)
    while True:
        image = cap.read()[1]
        image = cv.resize(image,None,fx = 0.75,fy = 0.75,interpolation = cv.INTER_AREA) #对图片进行缩放，INTER_AREA插值方法，这种插值方法比线性插值效果更好，更连续
        cv.imshow('VideoCapture', image)
        #33针速率，表示一秒钟有33张图片(1s = 1000ms , 1000/33 = 33),如果waiKey = 20 表示每秒有50帧的画面连续效果，这样的视频效果更好，但是消耗系统资源更多
        if(cv.waitKey(33) == 13): #如果用户没有按键，则过33毫秒自动退出,当用户回车键则退出
            break
    cap.release()  # 释放视屏捕捉设备资源
    cv.destroyAllWindows()  # 关闭cv 打开的隐藏窗口

captureVideo()
# captureImage()
