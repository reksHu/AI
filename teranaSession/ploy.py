#多项式模型
import platform
import numpy as np
import sklearn.pipeline as si
import sklearn.linear_model as sl
import sklearn.preprocessing as sp
import sklearn.metrics as sm
import matplotlib.pyplot as plt
# import matplotlib.patches as mc
import math
def generateData(x):
    if(x%3==0):
        return math.pow(x,2) + 20
    elif(x%2 == 0):
        return math.pow(x,2)/2
    elif(x%5==0):
        return math.pow(x, 2)/5
    else:
        return math.pow(x,2)
#绘制线性方程 y = pow(x,2)

def read_data():
    X = np.array(range(0,20))
    y =np.array([generateData(x) for x in X])
    return X.reshape(-1,1),y

#degree为多项式的次数
def train_model(degree,x, y):
    polyFeature = sp.PolynomialFeatures(degree)  # 多项式特性
    model = si.make_pipeline(polyFeature,sl.LinearRegression())
    model.fit(x, y)
    return model


def pred_model(model, x):
    y = model.predict(x)
    return  y

def eval_model(y, pred_y):
    mae = sm.mean_absolute_error(y,pred_y) #所有数据点的误差绝对值的平均数
    mse = sm.mean_squared_error(y, pred_y) #所有数据点的误差平方值得平均数
    mde = sm.median_absolute_error(y, pred_y) # 数据集中位数误差的绝对值中位数,有利于排除某些奇异值(Outlier干扰)
    evs = sm.explained_variance_score(y, pred_y)
    r2 = sm.r2_score(y, pred_y)

def init_chart():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.title("Polynomial Regression",fontsize = 20)
    plt.xlabel("x",fontsize = 14)
    plt.ylabel("y",fontsize = 14)
    plt.tick_params(which ="both",top=True,left=True,labelleft = True,  labelsize = 10)
    plt.grid(linestyle=':')


#绘制训练数据
def draw_train(train_x, train_y,pred_train_y):
    plt.plot(train_x,train_y,'s',c = 'limegreen')
    sorted_indices = train_x.T[0].argsort()
    plt.plot(train_x.T[0][sorted_indices],pred_train_y[sorted_indices],'--',
             c = 'lightskyblue')
    plt.legend(['Training','Predicted Training'],loc = 'upper left')
    # plt.legend()
#绘制测试数据
def draw_test(test_x, test_y, pred_text_y):
    plt.plot(test_x,test_y,'s',c = 'orangered',lable = 'Testing')
    plt.plot(test_x,pred_text_y,'o', c = 'dodgerblue', lable='Predict Testing')
    for x,pred_y, y in zip(test_x,pred_text_y,test_y):
        plt.gca().add_patch(plt.Arrow(x,pred_y,0,y-pred_y,width=0.8,ec ='none',fc='pink'))
    plt.legend()

def show_chart():
    # mng = plt.get_current_fig_manager()
    # if('Windows' in platform.system()):
    #     mng.window.state('zoomed')
    # else:
    #     mng.resize(*mng.window.maxsize())
    plt.show()
#绘制原始数据散点图
def draw_source(X, y ):
    plt.scatter(X, y, c = 'blue', s=30,marker='^')
    # plt.legend(['Souce Data'], loc = 'upper right')
    plt.show()

X, y = read_data()
# draw_source(X, y )
# init_chart()
model = train_model(2, X, y)
pred_y = pred_model(model,X)

draw_train(X,y,pred_y)

show_chart()