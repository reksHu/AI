#kmeans
import  pandas as pd
import numpy as np
import sklearn.cluster as sc
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pdb
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
    model = sc.KMeans( init='k-means++',n_clusters=4,n_init=10)
    model.fit(x)
    return model

def pred_model(model,x):
    y = model.predict(x)
    return y


def init_chart():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.title('Kmeans Cluster',fontsize = 20)
    plt.xlabel('x',fontsize = 14)
    plt.ylabel('y',fontsize = 14)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator()) #主坐标，刻度为1
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5)) #辅坐标，刻度为0.5
    ax.yaxis.set_major_locator(plt.MultipleLocator())
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))

    plt.tick_params(which='both',top = True, right = True, labelright = True, labelsize = 10)


def draw_grid(grid_x, grid_y ):
    plt.pcolormesh(grid_x[0],grid_x[1],grid_y,cmap = 'brg')
    plt.xlim(grid_x[0].min(),grid_x[0].max())
    plt.ylim(grid_x[1].min(),grid_x[1].max())

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
    model = model_train(x)
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