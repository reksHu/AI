import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as mg
def show_img(title,img):
    cv.imshow(title,img)

def read_img(fileName):
    orignal = cv.imread(fileName)
    return orignal

def calc_features(image):
    star = cv.xfeatures2d.StarDetector_create()  #利用star检测器处理彩色图片
    keypoints = star.detect(image)
    sift = cv.xfeatures2d.SIFT_create() #利用SIFT检测器处理灰度图
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    keypoints, desc = sift.compute(gray,keypoints)  #desc 为图片的特征值
    print(desc,image.shape)
    return desc

def draw_desc(desc):
    ma = plt.matshow(desc,cmap = 'jet')
    plt.gcf().set_facecolor(np.ones(3)*244/250)
    plt.title("DESC",fontsize = 20)
    plt.xlabel('Feature',fontsize = 14)
    plt.ylabel('Samples',fontsize = 14)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.xaxis.set_major_locator(plt.MultipleLocator())
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_minor_locator(plt.MultipleLocator())
    plt.tick_params(which = 'both',top = True, right = True, labeltop = False,labelbottom = True, labelsize = 10)
    dv = mg.make_axes_locatable(ax)
    ca = dv.append_axes('right','3%',pad='3%')
    cb = plt.colorbar(ma,cax=ca)
    cb.set_label('DESC',fontsize = 14)

def show_chart():
    plt.show()

def main():
    fileName =r"E:\python\study\terana_Learning\AI\data\penguin.jpg"
    img = read_img(fileName)
    show_img("Orignal",img)
    desc = calc_features(img)
    draw_desc(desc)
    show_chart()
    # cv.waitKey()

main()