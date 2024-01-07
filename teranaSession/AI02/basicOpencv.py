import cv2 as cv
import  numpy as np

img = cv.imread(r'E:\python\study\terana_Learning\AI\data\forest.jpg')
print(img.shape)
print(img)
cv.imshow('Original',img)


# 裁取图片部分,图片坐标以图片的左上角为原点
height, weight = img.shape[:2]
left, top = int(height/4),int(weight/4)
right, botton = int(height*3/4),int(weight*3/4)
cropped = img[top:botton,left:right]
cv.imshow('cropped',cropped)

blue,green,red = np.zeros_like(cropped),np.zeros_like(cropped),np.zeros_like(cropped)
blue[...,0] = cropped[...,0] #第0列，表示蓝色通道
green[...,1] = cropped[...,1] #第一列，绿色通道
red[...,2] = cropped[...,2] #第三列，红丝通道

cv.imshow('blue',blue)
cv.imshow('green',green)
cv.imshow('Red',red)

#图片缩放，放大
scaled = cv.resize(cropped,(weight,height),interpolation=cv.INTER_LINEAR) #interpolation=cv.INTER_LINEAR表示缩放后颜色像素填充算法
cv.imshow("Scaled",scaled)

#图片不按照等比例缩放
reformed = cv.resize(cropped,None,fx = 2,fy=0.5,interpolation = cv.INTER_LINEAR) #对图片横轴扩大2倍，纵向缩小2倍
cv.imshow("reformed",reformed)


cv.waitKey()