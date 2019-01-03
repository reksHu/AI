###学习曲线
import platform
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import sklearn.ensemble as se
import  numpy as np
import sklearn.preprocessing as sp
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
    return x, y

def eval_lv_model(model,x,y,train_sizes):
    train_sizes , train_scores, test_scores = ms.learning_curve(model,x,y,train_sizes=train_sizes,cv=5)
    return train_sizes,train_scores, test_scores

def get_model():
    model = se.RandomForestClassifier(n_estimators=200,max_depth=8,random_state=7)
    return model

def init_diagram():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.subplot(121)
    plt.title("Learning Curve",fontsize = 20)
    plt.xlabel("Number of Traning Sample",fontsize = 14)
    plt.ylabel("Accuracy",fontsize = 14)
    axis = plt.gca()
    axis.xaxis.set_major_locator(plt.MultipleLocator(100))
    axis.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.tick_params(which='both',right = True,top = True, labelright = True, labelsize = 10)
    plt.grid(linestyle = ":")

def draw_train_score(train_sizes,train_scores):
    plt.plot(train_sizes, train_scores.mean(axis = 1)*100,'o-',c = 'dodgerblue', label = 'Train Score')
    plt.legend()

def init_test_diag():
    plt.subplot(122)
    plt.title("Learning Curve", fontsize=20)
    plt.xlabel("Number of Traning Sample", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    axis = plt.gca()
    axis.xaxis.set_major_locator(plt.MultipleLocator(100))
    axis.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    plt.tick_params(which='both', right=True, top=True, labelright=True, labelsize=10)
    plt.grid(linestyle=":")

def draw_test_score(trainsizes, test_scores):
    plt.plot(trainsizes,test_scores.mean(axis = 1)*100,'o-',c='orange',label = 'Test Score')
    plt.legend()

def show_diag():
    plt.show()

def main():
    fileName = "./data/car.txt"
    x, y = read_data(fileName)
    model = get_model()
    train_sizes = np.linspace(100,1000,10).astype(int)
    train_sizes, train_scores, test_scores = eval_lv_model(model,x,y,train_sizes)
    print(train_scores.mean(axis = 1)*100)
    init_diagram()
    draw_train_score(train_sizes,train_scores)

    init_test_diag()
    draw_test_score(train_sizes,test_scores)
    show_diag()

if __name__ == '__main__':
    main()

