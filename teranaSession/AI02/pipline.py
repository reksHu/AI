import numpy as np
import  sklearn.datasets as sd
import sklearn.feature_selection as sf
import sklearn.ensemble as se
import sklearn.pipeline as sp
import sklearn.model_selection as sm
import matplotlib.pyplot as plt

def generate_samples():
    #n_informative: 4个类， n_features: 20 个特性
    x , y = sd.samples_generator.make_classification(n_informative=4, n_features= 20,n_redundant=0,random_state= 5, n_samples=100) #
    print(x.shape, y.shape)
    print(x[0])
    print(set(y))
    return x, y

def model_train(x, y ):
    skb = sf.SelectKBest(sf.f_regression,k=20) #选择K个最好的特性列做回归
    sfc = se.RandomForestClassifier(n_estimators = 25,max_depth = 5) #25个评估器

    model = sp.Pipeline([('mySelector',skb),('myClassifier',sfc)])
    # scores = sm.cross_val_score(model,x, y,  cv = 10, scoring='f1_weighted')
    # print(scores.mean())
    return  model

def cross_verfication(model,x,y):
    scores = sm.cross_val_score(model,x,y, cv = 10, scoring='f1_weighted')
    print("Scores: ",scores.mean())

def model_update(model,x, y):
    model.fit(x, y )
    #修改管道中的模型参数
    model.set_params(mySelector__k=2,myClassifier__n_estimators=20)
    # scores = sm.cross_val_score(model, x, y, cv=10, scoring='f1_weighted')
    # print(scores.mean())
    return model

def model_fit(model,x,y):
    model.fit(x,y)


def get_best_features(model,x):
    print(x.shape)
    selected_mask = model.named_steps['mySelector'].get_support() #获取被模型中最好的特性的掩码,返回值中有两个True,表示对应的两个最好的特性 mySelector__k=2
    print(selected_mask)

    # selected_mask =  np.array(selected_mask)
    # selected_index = np.argwhere(selected_mask==True)

    selected_index = np.arange(x.shape[1])[selected_mask]
    # print(selected_index)
    print("selected_index:",selected_index)

    x=x[:,selected_index]
    return x
    # model.fit(x, y )

def pred_model(model,x):
    pred_y = model.predict(x)
    return pred_y

def init_chart():
    plt.gcf().set_facecolor(np.ones(3)*240/255)  #背景为浅灰色
    plt.title("Pipleline Model",fontsize = 20)
    plt.xlabel('X',fontsize = 14)
    plt.ylabel('Y',fontsize = 14)
    # axs = plt.gca()
    # axs.xaxis.set_major_locator(plt.MultipleLocator(1))
    # plt.tick_params(axis='both',top = True , right = True,lableright = True, labelsize = 10)
    plt.tick_params(which='both', top=True, right=True, labelright=True, labelsize=10)
    plt.grid(True)

def draw_grid(grid_x,grid_y):
    plt.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='Dark2')
    plt.xlim(grid_x[0].min(), grid_x[0].max())
    plt.ylim(grid_x[1].min(), grid_x[1].max())

def draw_data(x, y ):
    plt.scatter(x[:,0],x[:,1],c = y,cmap='cool',s = 80)

def show_chart():
    plt.show()

def main():
    x, y = generate_samples()
    model = model_train(x, y)
    print("before model update score:",end='  ')
    cross_verfication(model,x,y)
    print("after model update score:", end='  ')
    model = model_update(model,x, y)
    cross_verfication(model,x, y )

    x = get_best_features(model,x)

    model.fit(x, y)
    cross_verfication(model,x,y)

    # 定义边界, l:x轴左边界，r:x轴右边界: h :水平方向步长
    l, r, h = x[:, 0].min() - 1, x[:, 0].max() + 1, 0.005
    # b: y轴底边界，t: 垂直方向顶边界，v :垂直方向步长
    b, t, v = x[:, 1].min() - 1, x[:, 1].max(), 0.005

    grid_x = np.meshgrid(np.arange(l, r, h), np.arange(b, t, v))  # 网格化,生成一个二维数组，分别对应平面上x和y坐标
    flat_x = np.c_[grid_x[0].ravel(),grid_x[0].ravel()] #按列合并生成x,y轴数据
    # grid_y = pred_model(model, np.c_[grid_x[0].ravel(), grid_x[1].ravel()]).reshape(grid_x[0].shape)  # 对grid区域进行预测
    flat_y = pred_model(model,flat_x)

    grid_y = flat_y.reshape(grid_x[0].shape)
    # prd_y = pred_model(model,x)

    init_chart()
    draw_grid(grid_x,grid_y)
    draw_data(x,y)
    show_chart()

main()






















