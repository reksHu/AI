import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as sn

def make_data():
    x = 10 * np.random.rand(100,1) - 5 # (100,1)的数列
    y = np.sinc(x).ravel() # 采用 sin(x)/x的函数
    y += 0.2*(0.5-np.random.rand(y.size)) #制作噪声
    print(x[:10])
    print(y[:10])
    return x,y

def test_data():
    x = np.linspace(-5,5,10000).reshape(-1,1) #产生10000个测试数据
    return x

def model_train(x, y):
    model = sn.KNeighborsRegressor(n_neighbors=10,weights='distance',n_jobs = 2)
    model.fit(x, y)
    return model

def pred_model(model,x):
    pred_y = model.predict(x)
    return pred_y

def init_chart():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.title("KNN Regration",fontsize = 20)
    plt.xlabel("X",fontsize = 14)
    plt.ylabel("y",fontsize = 14)
    axs = plt.gca()
    axs.xaxis.set_major_locator(plt.MultipleLocator(1))
    axs.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    axs.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    axs.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.tick_params(axis='both',top = True,right = True ,labelright = True,labelsize  = 10)
    plt.grid(linestyle = ":")

def draw_data(x,y,test_x,pred_y):
    sorted_index = x.ravel().argsort()
    plt.plot(x[sorted_index],y[sorted_index],'o-',c = 'dodgerblue',label = 'Training')
    plt.plot(test_x,pred_y, c = 'orange',label = 'Testing')
    plt.legend()

def show_chart():
    plt.show()

def main():
    x, y = make_data()
    model = model_train(x,y)

    test_x = test_data()
    pred_y = pred_model(model,test_x)

    init_chart()
    draw_data(x, y,test_x,pred_y)
    show_chart()

main()