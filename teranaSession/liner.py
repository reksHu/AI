import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import  numpy as np
import sklearn.metrics as sm
iris = load_iris()
#绘制散点图
def drawScatterDiagram():
    colors = ['blue', 'red', 'green']
    x_index,y_index = 0,1
    zips = zip(range(len(iris.target_names)), colors)
    for lable, color in zips:
        plt.scatter(iris.data[iris.target==lable,x_index],
                    iris.data[iris.target==lable,y_index],
                    c = color)
        plt.legend(iris.target_names, loc='upper right')
    plt.xlabel(iris.feature_names[x_index])
    plt.ylabel(iris.feature_names[y_index])

    plt.show()

def generateData(x):
    if(x%3==0):
        return 0.9*x + 0.6
    elif(x%2 == 0):
        return 0.5*x+0.3
    else:
        return 0.8*x + 0.7
#绘制线性方程 y = 0.8x + 0.7
def getLiner():

    X = np.array(range(-5,20))
    y =np.array([generateData(x) for x in X])
    # plt.scatter(X,y,c = 'dodgerblue', s=60)
    # plt.legend(['Source Data Scatter'],loc = 'upper left')
    # plt.xlabel('X')
    # plt.ylabel('y')
    # plt.show()
    # print(y)
    return X.reshape(-1,1), y

def linerPredect(X,y):
    import sklearn.linear_model as lm
    model = lm.LinearRegression()
    model.fit(X,y)
    pred_y = model.predict(X)
    print("pred_y:",pred_y)
    plt.scatter(X,y,c='blue',s=30, marker='^')
    plt.scatter(X,pred_y,c='green',s = 50)
    plt.legend(['Source Data','Predict Data'],loc = 'upper left')
    plt.show()
    print("y:",y)

    mae = sm.mean_absolute_error(y,pred_y)
    print(mae)
    print("r2:",sm.r2_score(y,pred_y))
    return pred_y
# drawScatterDiagram()

def readFile():
    filePath = "data.txt"
    # 读取训练数据集
    x, y = [], []
    with open(filePath, 'r') as f:
        for line in f.readlines():
            data = [float(substr) for substr in
                    line.split(',')]
            x.append(data[:-1])
            y.append(data[-1])
    x = np.array(x)
    y = np.array(y)
    print(x,y)

def drawtrain(train_x,train_y,pred_train_y1):
    plt.plot(train_x,train_y,'x',c='limegreen')
    sorted_indices = train_x.T[0].argsort()
    plt.plot(train_x.T[0][sorted_indices],pred_train_y1[sorted_indices],'--',c='lightskyblue')
    plt.legend(['Training','Predicated Training'],loc='upper left')
    plt.show()

def drawTest(text_x,text_y,pred_test_y1):
    plt.plot(text_x,text_y,'s',c='orangered')
    plt.plot(text_x,pred_test_y1,'o',c='lightskyblue')
    for x,pred_y , y in zip(text_x,pred_test_y1,text_y):
        plt.gca().add_patch(plt.Arrow(x,pred_y,0,y-pred_y,width=0.8,ec = "none",fc='pink'))
    plt.legend(['Testing','Predicat Testing'])
    plt.show()

X, y = getLiner()
pred_y = linerPredect(X,y)
# drawtrain(X,y,pred_y)
drawTest(X,y,pred_y)
# linerPredect(X,y)