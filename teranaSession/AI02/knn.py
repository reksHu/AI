import numpy as np
import sklearn.neighbors as sn
import matplotlib.pyplot as plt

def read_data(fileName):
    with open(fileName,'r') as f:
        lines = f.readlines()
        x , y = [],[]
        data = []
        for line in lines:
            line_data = line[:-1]
            line_data = [float(d) for d in line_data.split(',')]
            data.append(line_data)

    data = np.array(data).T
    for row in range(len(data)):
        if(row< len(data)-1):
            x.append(data[row])
        else:
            y.append(data[row])
    x = np.array(x).T
    y = np.array(y,dtype=int)
    y = y[0]
    return  x, y

def read_date2(fileName):
    with open(fileName,'r') as f:
        lines = f.readlines()
        data= []
        x,y = [],[]
        for line in lines:
            data = [float(d) for d in line.split(',')]
            x.append(data[:-1])
            y.append(data[-1])

    return np.array(x), np.array(y,dtype=int)

def test_data():
    test_x = [
        [2.82,4.4]
        ]
    return np.array(test_x)

def model_train(x, y ):
    model = sn.KNeighborsClassifier(n_neighbors=10,weights='distance')
    model.fit(x, y )

    return model

def predict_model(model,x):
    pred_y = model.predict(x)
    return pred_y

def get_neihbors(model,x):
    nn_instance, nn_indexes = model.kneighbors(x)
    return nn_indexes,nn_indexes

def init_chart():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.title('K Neighbors Classifier',fontsize = 20)
    plt.xlabel('X',fontsize = 14)
    plt.ylabel('Y',fontsize = 14)
    plt.tick_params(axis='both',top = True,right = True, labelright = True, labelsize = 10)
    plt.grid(linestyle = ':')

def draw_data(train_x, train_y,test_x,test_y,nn_indexse):
    classes = np.unique(train_y)
    classes.sort()
    cs = plt.get_cmap('gist_rainbow',len(classes))(range(len(classes)))
    plt.scatter(train_x[:,0],train_x[:,1],c=cs[train_y], s = 60)
    plt.scatter(test_x[:,0],test_x[:,1],c=cs[test_y],marker='D',s = 120)

    for i, nn_index in enumerate(nn_indexse): #绘制与预测值临近的点
        plt.scatter(train_x[nn_index,0],train_x[nn_index,1],marker='D',edgecolors=cs[np.ones_like(nn_index) * test_y[i]],
                    facecolor = 'none',linewidths=2,s = 100)



def draw_test_data(test_x,test_y):
    plt.scatter(test_x[:,0],test_x[:,1],c=test_y, s=80,marker='+')

def draw_grid(grid_x,grid_y):
    plt.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='brg')
    plt.xlim(grid_x[0].min(), grid_x[0].max())
    plt.ylim(grid_x[1].min(), grid_x[1].max())

def show_chart():
    plt.show()

def main():
    fileName = r"D:\python\study\terana_Learning\AI\data\knn.txt"
    train_x, train_y = read_data(fileName)
    model = model_train(train_x, train_y)
    l,r, h = train_x[:,0].min()-1,train_x[:,0].max()+1, 0.005
    b, t, v = train_x[:,1].min()-1,train_x[:,1].max()+1 , 0.005
    grid_x = np.meshgrid(np.arange(l,r,h),np.arange(b,t,v))
    flat_x = np.c_[grid_x[0].ravel(),grid_x[1].ravel()]
    grid_y = predict_model(model,flat_x).reshape(grid_x[0].shape)

    test_x = test_data()
    pred_y = predict_model(model,test_x)
    nn_indexes, nn_indexes = get_neihbors(model,test_x)
    init_chart()
    draw_grid(grid_x,grid_y)
    draw_data(train_x,train_y,test_x,pred_y,nn_indexes)

    show_chart()

main()