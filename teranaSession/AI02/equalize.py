import cv2 as cv

img = cv.imread(r'E:\python\study\terana_Learning\AI\data\sunrise.jpg')

cv.imshow('Oraginal',img)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #conver to color，将RGB颜色转换成灰度图
cv.imshow("Gray", gray)

equalized_gry = cv.equalizeHist(gray) #对灰度图进行均衡化,均衡化后的结果使得灰度图适当部分颜色增亮
cv.imshow("equalized", equalized_gry)

#yuv 是一种亮度，色度和饱和度构成的颜色通道, y :亮度通道，U，色度通道，V:饱和度通道
#yuv不能直接用imshow显示，imshow只能显示RGB颜色通道的图片
yuv = cv.cvtColor(img,cv.COLOR_BGR2YUV)

#将亮度通道取出，再饱和化
yuv[...,0] = cv.equalizeHist(yuv[...,0])
equalize_rgb = cv.cvtColor(yuv,cv.COLOR_YUV2BGR)
cv.imshow('RGB',equalize_rgb)
cv.waitKey()