#验证曲线
import  numpy as np
import  matplotlib.pyplot as plt
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms

def read_data(filename):
    data = []
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(line[:-1].split(','))  #去掉最后的换行符
    data = np.array(data).T #将行转为列为下一步编码准备，在编码过程中行比列好处理
    encoders, x = [], []
    y = None
    for row in range(len(data)):
        encoder = sp.LabelEncoder()
        if(row < len(data)-1):
            xencoder = encoder.fit_transform(data[row])
            x.append(xencoder)
        else:
            y = encoder.fit_transform(data[row]) #此处返回整个数组,所以在之前不用定义y
        encoders.append(encoder)
    x = np.array(x).T
    print(x.shape,type(x),type(y),y.shape)
    return x, y, encoders

def train_model(x,y):
    model = se.RandomForestClassifier(max_depth=8,n_estimators=200,random_state=7)
    model.fit(x,y)
    return model

def train_model_estimator(max_depth):
    model = se.RandomForestClassifier(max_depth=max_depth,random_state=7)
    return model

def eval_vc_estimator(model,x,y,n_estimators):
    train_scores,test_scores =  ms.validation_curve(model,x,y,'n_estimators',n_estimators,cv = 5) # 5次交叉验证，得出的结果集为5列
    return train_scores,test_scores

def train_model_max_depth(n_estimators):
    model = se.RandomForestClassifier(n_estimators = n_estimators,random_state=7)
    return model

def eval_vs_max_depth(model,x,y,maxdepth):
    train_scores, test_scores =  ms.validation_curve(model,x,y,'max_depth',maxdepth,cv = 5) #
    return train_scores,test_scores


#检验模型精度
def eval_ac(y, pred_y):
    ac = ((y==pred_y).sum() / pred_y.size)
    print("Accuracy: {}%".format(round(ac*100,2)))

def init_estimator():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.subplot(121)
    plt.title("Training Curve On Estimators",fontsize = 20)
    plt.xlabel("Number of Estimators",fontsize = 14)
    plt.ylabel("Accuracy",fontsize = 14)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(20)) #设置x轴主刻度线为20
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1)) #设置y轴单位主刻度为0.1
    plt.tick_params(which = 'both',top = True, right = True , labelright = True, labelsize = 10)
    plt.grid(linestyle = ":")

def draw_estimators(n_estimators,train_score_estimators):
    plt.plot(n_estimators,
             train_score_estimators.mean(axis = 1) * 100,'o-',c='dodgerblue',label = 'Train Score') # train_score_estimators为10列5行数组，由于每一列为一个验证器生成的值，所以根据每列求均值
    plt.legend() #显示出图例

def init_max_deep_diagram():
    plt.subplot(122)
    plt.title("Training Curve On Max Depth",fontsize = 20)
    plt.xlabel("Number of MaxDepth", fontsize = 14)
    plt.ylabel("Accuracy",fontsize = 14)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    plt.tick_params(which = 'both',top = True,right = True,labelright = True, labelsize = 10)
    plt.grid(linestyle = ":")

def draw_max_depth(n_max_depth,train_score_max_depth):
    plt.plot(n_max_depth,train_score_max_depth.mean(axis=1)*100,'o-',c='orange',label = "Train Score")
    plt.legend()


def show_chart():
    plt.show()

def main():
    fileName = "./data/car.txt"
    train_x, train_y, encoders = read_data(fileName)
    model_estimator = train_model_estimator(4)
    n_estimators = np.linspace(20,200,10).astype(int)
    train_score_estimator, test_score_estimator =  eval_vc_estimator(model_estimator,train_x,train_y,n_estimators)
    print(train_score_estimator,test_score_estimator)

    model_max_depth = train_model_max_depth(20)
    n_max_depth = np.linspace(1,10,10).astype(int)
    train_score_max_depth ,test_score_max_depth = eval_vs_max_depth(model_max_depth,train_x,train_y,n_max_depth)

    init_estimator()
    draw_estimators(n_estimators,train_score_estimator)

    init_max_deep_diagram()
    draw_max_depth(n_max_depth,train_score_max_depth)
    show_chart()

if __name__ == '__main__':
    main()