#寻找最近邻
import numpy as np
import sklearn.neighbors as sn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

def make_data():
    x = [
        [3.0, 7.1],
        [2.3, 9.2],
        [1.7, 9.0],
        [1.0, 7.0],
        [1.0, 5.0],
        [1.7, 3.5],
        [4.3, 3.4],
        [5.0,5.9],
        [5.0, 7.1],
        [4.3, 9.0],
        [3.7, 9.0],
        [2.8, 5.9]
    ]
    x = np.array(x)
    print(x[0])
    return x

def test_data():
    test_x  = [
        [4.5,4.5]
    ]

    return np.array(test_x)

def train_model(x):
    model = sn.NearestNeighbors(n_neighbors=3,algorithm='ball_tree')
    model.fit(x)
    return model

def model_predict(model,x):
    nn_distance,nn_index  = model.kneighbors(x)
    return nn_distance, nn_index

def init_chart():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.title('Nearest Neighbor',fontsize = 20)
    plt.xlabel('X',fontsize = 14)
    plt.ylabel('Y',fontsize = 14)
    axis = plt.gca()
    axis.xaxis.set_major_locator(plt.MultipleLocator())
    axis.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    axis.yaxis.set_major_locator(plt.MultipleLocator())
    axis.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    plt.tick_params(axis='both',top = True,labelsize = 10)
    plt.grid(linestyle=":")

def draw_train_data(x):
    plt.scatter(x[:,0],x[:,1],c = 'k',s = 80,zorder = 2)


def draw_test_data(test_x,train_x,nn_index):
    cs = plt.get_cmap('gist_rainbow',len(nn_index))(range(len(nn_index)))
    for i ,nn_index in enumerate(nn_index):
        plt.gca().add_patch(mpatch.Polygon(
            train_x[nn_index],edgecolor='none',facecolor=cs[i],alpha = 0.2,zorder = 0
        ))   #获取当前的图，并加一个多变形
        plt.scatter(test_x[i,0],test_x[i,1],c=cs[i],s = 80,zorder = 1)


def show_chart():
    plt.show()


def main():
    train_x = make_data()
    test_x = test_data()
    model = train_model(train_x)
    nn_distance, nn_index = model_predict(model,test_x)
    print(nn_distance,nn_index )
    init_chart()
    draw_train_data(train_x)
    draw_test_data(test_x,train_x,nn_index)
    show_chart()

main()