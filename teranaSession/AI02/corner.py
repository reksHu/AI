import cv2 as cv

img_path =r'E:\python\study\terana_Learning\AI\data\box.png'

img = cv.imread(img_path)
cv.imshow("Original",img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Gray",gray)

corner = cv.cornerHarris(gray,7,5,0.04)  #Harris棱角检测器,7,5为卷积和, 得到棱角可能点，为浮点数
corner = cv.dilate(corner,None)  #由于检测出来的棱角点太小，将其方法方便作图观察
threadhold = corner.max()*0.01 #设置棱角点的阀值, 如果值越小，被检测为棱角点的概率就高,即被标记为棱角点的地方就越多
print(threadhold)
corner_mask = corner > threadhold
img[corner_mask] = [0,0,255] # BRG颜色值，将棱角点显示为红色

cv.imshow('Result',img)
cv.waitKey()