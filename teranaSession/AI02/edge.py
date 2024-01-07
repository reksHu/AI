import cv2 as cv

img = cv.imread(r'E:\python\study\terana_Learning\AI\data\box.png')
sobel = cv.Sobel(img,cv.CV_64F,1,0,ksize = 5)
cv.imshow('source',img)
cv.imshow('sobel',sobel)


lap = cv.Laplacian(img,cv.CV_64F)
cv.imshow('lap',lap)
canny = cv.Canny(img,50,240)
cv.imshow("canny",canny)

cv.waitKey()
