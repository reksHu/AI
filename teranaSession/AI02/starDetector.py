# Star图像检测器
import cv2 as cv
img_path = r'E:\python\study\terana_Learning\AI\data\table.jpg'
def star_detect(img_path):
    img = cv.imread(img_path)
    cv.imshow("Original",img)

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cv.imshow("Gray",gray)
    detector = cv.xfeatures2d.StarDetector_create()
    keypoints = detector.detect(gray) #返回keypoints 对象，该对象包括检测点的位置和方向(矢量)，表示监测点的位置和该位置上点的颜色变化
    img2 = img
    cv.drawKeypoints(img, keypoints, img2,flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 将keypoints 的图保存在img中(第三个参数)
    cv.imshow("img",img2)  #图中圆圈越大表示权重越大
    print(keypoints)
    cv.waitKey()

def Sift_detect(img_path):
    img = cv.imread(img_path)
    cv.imshow("Original", img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Gray", gray)
    detector = cv.xfeatures2d.SIFT_create()
    keypoints = detector.detect(gray)  # 返回keypoints 对象，该对象包括检测点的位置和方向(矢量)，表示监测点的位置和该位置上点的颜色变化
    img2 = img
    cv.drawKeypoints(img, keypoints, img2,
                     flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 将keypoints 的图保存在img中(第三个参数)
    cv.imshow("img", img2)  # 图中圆圈越大表示权重越大
    print(keypoints)
    cv.waitKey()

Sift_detect(img_path)
# star_detect(img_path)