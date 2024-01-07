#实现均值漂移模型

import numpy as np
import pandas as pd
import sklearn.cluster as sc
import matplotlib.pyplot as plt

def read_data(file_name):
    with open(file_name,'r') as f:
        lines = f.readlines()
        x = []
        for line in lines:
            line_data = line[:-1].split(',')
            data = [float(d) for d in line_data]
            x.append(data)

    return np.array(x)

def train_model(x):
    # 区域边侧检测， bw确定概率密度之间的宽度，用来定义边界，宽度区域越宅，表示越精确
    bw = sc.estimate_bandwidth(x,n_samples=len(x),quantile=0.1) #定义评估器带宽,n_samples 样本个数 quantile：量化因子,两个点之间平均距离大概值，可以慢慢增加用于找出不同类别的边界
    model = sc.MeanShift(bandwidth=bw , bin_seeding = True) #均值漂移模型, bin_seeding,二进制起始种子，模型自己制定，模型将在开始时自动指定一个初始化种子
    model.fit(x)
    return model

def pred_model(model,x):
    y = model.predict(x)
    return y

def init_chart():
    plt.gcf().set_facecolor(np.ones(3)*204/255)
    plt.title("Means Shift",fontsize = 20)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator())  # 主坐标，刻度为1
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))  # 辅坐标，刻度为0.5
    ax.yaxis.set_major_locator(plt.MultipleLocator())
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))

    plt.tick_params(which='both', top=True, right=True, labelright=True, labelsize=10)


def draw_grid(grid_x, grid_y ):
    plt.pcolormesh(grid_x[0],grid_x[1],grid_y,cmap = 'brg')
    plt.xlim(grid_x[0].min(),grid_x[0].max())
    plt.ylim(grid_y[1].min(),grid_y[1].max())

def draw_data(x,y):
    plt.scatter(x[:,0],x[:,1],c=y,cmap='RdYlBu', s=80)

def show_chart():
    plt.show()

def draw_centers(centers):
    plt.scatter(centers[:,0],centers[:,1],c = 'black',marker='+',s = 1000,linewidths=1) # linewidths 表示线宽

def main():
    fileName =r"E:\python\study\terana_Learning\AI\data\multiple3.txt"
    x =  read_data(fileName)
    l,r,h = x[:,0].min()-1,x[:,0].max()+1, 0.005
    b,t,v = x[:,1].min()-1, x[:,1].max()+1 , 0.005
    model = train_model(x)
    grid_x = np.meshgrid(np.arange(l,r,h), np.arange(b,t,v))
    grid_y = pred_model(model,np.c_[grid_x[0].ravel(),grid_x[1].ravel()]).reshape(grid_x[0].shape) #对grid区域进行预测

    pred_y = pred_model(model,x)

    init_chart()
    draw_grid(grid_x,grid_y)
    draw_data(x,pred_y)
    print(model.cluster_centers_)
    draw_centers(model.cluster_centers_)
    show_chart()
    return 0

if __name__ == '__main__':
    main()