#朴素贝叶斯分类器, 作图的过程中有错误
from __future__ import unicode_literals
import sklearn.naive_bayes as nb
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms

def read_data():
    file = "./data/multiple1.txt"
    with open(file,'r', encoding='utf-8') as f:
        lines = f.readlines()
        x, y  = [],[]
        for line in lines:
            data = [float(d) for d in line.split(',')]
            x.append(data[:-1])
            y.append(data[-1:])

        return np.array(x), np.array(y,dtype=int)

def train_data(x, y ):
    model = nb.GaussianNB()
    model.fit(x, y)
    return model

def pred_model(model,x):
    y = model.predict(x)

    return  y

def eval_accu(y, pred_y):
    accurateRate = (y==pred_y).sum()/pred_y.size()
    print("{}%".format(round(accurateRate*100,2)))

def init_chart():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.title("Naive Bayes Classifier",fontsize = 20)
    plt.xlabel('x',fontsize = 12)
    plt.ylabel('y',fontsize = 12)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator())
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(plt.MultipleLocator())
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    plt.tick_params(which='both',top=True,right = True,labelright =True,labelsize = 10)
    plt.grid(axis='y',linestyle = ':')

def draw_grid(grid_x, grid_y):
    plt.pcolormesh(grid_x[0],grid_x[1],grid_y,cmap='gray')
    plt.xlim(grid_x[0].min(),grid_x[0].max())
    plt.ylim(grid_x[1].min(),grid_x[1].max())

def draw_data(x, y):
    cVal = ['r','y','g','b']
    plt.scatter(x[:,0],x[:,1],c=y,cmap='RdYlBu',s=80)

def show_chart():
    plt.show()

def main():
    x, y = read_data()
    train_size = int(len(x)*0.8)
    train_x = x[:train_size]
    train_y = y[:train_size]
    l, r, h = train_x[:,0].min() -1,train_x[:,0].max()+1,0.005
    b,t,v = train_x[:,1].min() - 1,train_x[:,1].max() + 1,0.005
    model = train_data(train_x, train_y)
    grid_x = np.meshgrid(np.arange(l,r,h),np.arange(b,t,v))

    flat_x = np.c_[grid_x[0].ravel(), grid_x[1].ravel()]
    flat_y = pred_model(model,flat_x)
    grid_y = flat_y.reshape(grid_x[0].shape)
    # pred_y = pred_model(model,train_x)

    # grid_y = pred_model(model,np.c_[grid_x[0].ravel(),grid_x[1].ravel()]).reshape(grid_x[0].shape)
    init_chart()
    draw_grid(grid_x,grid_y)
    draw_data(train_x, train_y)
    show_chart()

if __name__ == '__main__':
    main()