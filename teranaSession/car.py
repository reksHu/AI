import numpy as np
import pandas as pd
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
    print(y[:5],type(y))
    return x, y, encoders

def train_model(x,y):
    model = se.RandomForestClassifier(max_depth=8,n_estimators=200,random_state=7)
    model.fit(x,y)
    return model

def eval_cv(model, x, y):
    pc = ms.cross_val_score(model,x,y,cv = 2,scoring='precision_weighted')
    rc = ms.cross_val_score(model, x, y, cv = 2, scoring='recall_weighted')
    f1 = ms.cross_val_score(model, x, y, cv = 2 ,scoring= 'f1_weighted')
    ac = ms.cross_val_score(model, x, y, cv = 2, scoring= 'accuracy')
    print("{}% {}% {}% {}%".format(round(pc.mean()*100,2) ,round(rc.mean()*10,2),
                                   round(f1.mean()*100,2), round(ac.mean()*100,2) ))

#制造人为测试数据
def make_data(encoders):
    data = [
        ['high','med','5more','4','big','low','unacc'],
        ['high','med','4','4','med','med','acc'],
        ['low','low','2','4','small','high','good'],
        ['low','med','3','4','med','high','vgood'],
        ['low', 'med', '5more', '4', 'med', 'high', 'vgood'],
        ['high', 'low', '2', '4', 'med', 'high', 'good'],
        ['low', 'med', '3', 'more', 'small', 'med', 'unacc']
    ]
    data = np.array(data).T
    x, y = [],[]
    for row in range(len(data)):
        encoder = encoders[row]
        if(row<len(data)-1):
            x.append(encoder.transform(data[row]))
        else:
            y.append(encoder.transform(data[row]))
    x = np.array(x).T
    return  x, y


def pred_model(model,x):
    pred_y = model.predict(x)
    return pred_y


#检验模型精度
def eval_ac(y, pred_y):
    ac = ((y==pred_y).sum() / pred_y.size)
    print("Accuracy: {}%".format(round(ac*100,2)))


def main():
    fileName = "./data/car.txt"
    train_x, train_y, encoders = read_data(fileName)
    model = train_model(train_x, train_y)
    eval_cv(model,train_x,train_y)

    test_x, test_y = make_data(encoders)
    pred_test_y = pred_model(model,test_x)
    eval_ac(test_y,pred_test_y)
    print("test：",encoders[-1].inverse_transform(test_y))
    print("pred：",encoders[-1].inverse_transform(pred_test_y))

    encoder_test_x =[]
    print("test x :",test_x, test_x.T)
    tran_test_x = test_x.T
    for index in range(len(encoders)-1):
       # print(encoders[index].inverse_transform(test_x[index]))
        print(encoders[index])
        encoder_test_x.append(encoders[index].inverse_transform(tran_test_x[index]))
    print(np.array(encoder_test_x).T)


if __name__ == '__main__':
    main()


