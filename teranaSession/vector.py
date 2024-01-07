import numpy as np
import sklearn.cluster as sc
import scipy.misc as sm
import matplotlib.pyplot as plt


def train_model(n_clusters,x):
    model = sc.KMeans(init='k-means++', n_clusters=n_clusters,n_init=4,random_state=5)
    model.fit(x)
    return model

def load_image(image_file):
    return sm.imread(image_file,True).astype(np.uint8) #使用一个字节表示图片字符，默认图片返回的是浮点型，因为矢量量化的类不方便使用浮点，所以装换成int

'''
bpp: binary per pix,每像素多少位，如果是8位，则2的8次方=256个数，每个数代表一个颜色，一共有256个颜色，则一个元素有256种颜色，数值越大表示的图像颜色越丰富
'''
def compress_image(image,bpp):
    n_cluster = np.power(2,bpp)
    x = image.reshape((-1,1)) #将image变换成一维数组，一列的一维数组,用于训练
    print(x)
    model = train_model(n_cluster,x)
    y = model.labels_ #获取分类后的类别的索引，从0开始整数
    print(y)
    # 得到每个中心点坐标和中心点类的索引号组成的数组,比如 第0号分类元素，center[0]得到了第0号分类元素的中心位置，
    # 这里的中心位置其实是相近的颜色值
    centers = model.cluster_centers_.squeeze()
    print(centers)
    z = centers[y] #获取压缩后图片的颜色值,现在z 为一维数组,实现矢量量化，用中心的颜色取代周围的颜色
    return  z.reshape(image.shape) #返回原来图片维度的压缩后的颜色值

def init_chart():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.title("Image Compress",fontsize = 20)
    plt.axis('off')

def draw_image(image):
    plt.imshow(image,cmap= 'gray') #将0-255的颜色值对应成灰度值

def show_chart():
    plt.show()

def main():
    image_file = r"E:\python\study\terana_Learning\AI\data\lily.jpg"
    # image_file = r"E:\python\study\terana_Learning\AI\data\table.jpg"
    image = load_image(image_file)
    # compressed_image = compress_image(image,8)#原始图片也是为每像素8位表示，所以这里没有实际上没有压缩
    # compressed_image = compress_image(image, 4) #压缩成 2^4 个颜色值，则一共有16中颜色,但是图片显示则使用灰度统一显示
    compressed_image = compress_image(image, 1) # 一位则表示两种颜色，黑和白
    init_chart()
    draw_image(compressed_image)
    show_chart()

main()