#凝聚层次
import  pandas as pd
import numpy as np
import sklearn.cluster as sc
import sklearn.metrics as sm
import platform
import matplotlib.pyplot as plt

def read_data(file_name):
    data, x = [], []
    with open(file_name,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line_data = line[:-1].split(',')
            data = [float(d) for d in line_data]
            x.append(data)
    return np.array(x)

def model_train(x):
    model = sc.AgglomerativeClustering(linkage='ward',n_clusters=4) #ward 表示对方向上收敛
    return model

def pred_model(model,x):
    y = model.fit_predict(x)
    return y


def init_chart():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.title('Hierarchical Agglomerative Cluster',fontsize = 20)
    plt.xlabel('x',fontsize = 14)
    plt.ylabel('y',fontsize = 14)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator()) #主坐标，刻度为1
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5)) #辅坐标，刻度为0.5
    ax.yaxis.set_major_locator(plt.MultipleLocator())
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))

    plt.tick_params(which='both',top = True, right = True, labelright = True, labelsize = 10)
    plt.grid(linestyle=":")

def draw_data(x,y):
    plt.scatter(x[:,0],x[:,1],c=y,cmap='RdYlBu', s=80)

def show_chart():
    mng = plt.get_current_fig_manager()
    # if('Windows' in platform.system()):
    #     mng.window.state('zoomed')
    # else:
    #     mng.resize(*mng.window.maxsize())
    plt.show()


def main():
    fileName =r"E:\python\study\terana_Learning\AI\data\multiple3.txt"
    x =  read_data(fileName)

    model = model_train(x)

    pred_y = pred_model(model,x)

    init_chart()

    draw_data(x,pred_y)

    show_chart()
    return 0



if __name__ == '__main__':
    main()