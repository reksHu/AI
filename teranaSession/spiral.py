#
import platform
import matplotlib.pyplot as plt
import  numpy as np
import sklearn.cluster as sc
import sklearn.neighbors as sn

#n_noise噪声， n_samples表示样本数量
#得到玫瑰线方程
def make_data(n_noise = 0.05,n_samples = 500):
    #np.random.rand(n_samples,1) 创建  shape = (500,1) ， 0-1的数组
    t = 2.5 * np.pi * ( 1 + 2 * np.random.rand(n_samples,1))
    x = 0.05 * t * np.cos(t)
    y = 0.05 * t * np.sin(t)
    n = n_noise * np.random.rand(n_samples,2)
    return np.hstack((x,y)) + n

#训练处不连续性的模型(根据数据点的距离分类)
def train_model_no(x):
    model = sc.AgglomerativeClustering(linkage='average',n_clusters = 3 ) #以均衡方式来分类
    return model

#训练处有连续性的模型，也就是根据数据点的连续性来分类
#connectivity= sn.kneighbors_graph(x,10,include_self=False) ,指定连续性的方式，通过连续性图找出10邻居点作为连续性基准点，不包括自己
def train_model_10(x):
    model = sc.AgglomerativeClustering(linkage='average',n_clusters= 3,connectivity= sn.kneighbors_graph(x,10,include_self=False))
    return model

def pred_model(model,x):
    return model.fit_predict(x)

#初始化没有连续性的图
def init_model_no():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.subplot(121) #一行两列第一个
    plt.title('Connectivity : No',fontsize = 20)
    plt.xlabel('x',fontsize = 14)
    plt.ylabel('y',fontsize = 14)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    plt.tick_params(which ='both', right = True, top = True, labelright = True,labelsize = 10)
    plt.grid(linestyle = ":")

#初始化连续性的图
def init_model_10():
    plt.subplot(122) #一行两列第一个
    plt.title('Connectivity : 10',fontsize = 20)
    plt.xlabel('x',fontsize = 14)
    plt.ylabel('y',fontsize = 14)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    plt.tick_params(which ='both', right = True, top = True, labelright = True,labelsize = 10)
    plt.grid(linestyle = ":")

def draw_chart(x, y ):
    plt.scatter(x[:,0],x[:,1],c = y,cmap='brg',s = 80)

def show_chart():
    plt.show()

def main():
    x = make_data()
    model_no = train_model_no(x)
    model_10 = train_model_10(x)
    pred_no = pred_model(model_no,x)
    pred_10 = pred_model(model_10,x)

    init_model_no()
    draw_chart(x,pred_no)

    init_model_10()
    draw_chart(x,pred_10)

    show_chart()
main()