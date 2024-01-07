import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.preprocessing as sp
import sklearn.svm as svm
import  sklearn.metrics as sm
from income.DigitEncoder import MyDigitEncoder
'''
根据比赛球赛预测交通车辆通行情况
星期     时间   比赛队伍    是否在比赛中  周围路口车辆
Tuesday,00:00,San Francisco,no,3
'''
def read_data(fileName):
    data = []
    x, y =[],[]
    with open(fileName,'r') as f:
        lines = f.readlines()
        for line in lines:
            line_data = line[:-1]
            data.append(line_data.split(','))
    data = np.array(data).T
    encoders = []
    for row in range(len(data)):
        if(data[row][0].isdigit()):
            encoder = MyDigitEncoder()
        else:
            encoder = sp.LabelEncoder()
        if(row<len(data)-1):
            x.append(encoder.fit_transform(data[row]))
        else:
            y = encoder.fit_transform(data[row])
        encoders.append(encoder)
    x = np.array(x).T
    return x, y,encoders

def train_model(x, y ):
    # model = svm.SVR(kernel='rbf',C = 10, epsilon = 0.2) # C 为惩罚值，spsilon 为惩罚系数
    model = svm.SVC(kernel='rbf',class_weight='balanced')
    model.fit(x, y)
    return  model

#交叉验证
def eval_cv(model,x,y):
    ac = ms.cross_val_score(model,x,y,cv = 3, scoring='accuracy')
    print("Accuracy:{}%".format(round(ac.mean()*100,2)))

#预测
def pred_model(model,x):
    pred_y = model.predict(x)
    return pred_y

#精度评估
def eval_ac(y,pred_y):
    ac = ((y==pred_y).sum()/pred_y.size)
    print("Accuracy:{}%".format(round(ac.mean() * 100, 2)))

#评估模型
def eval_model(y,pred_y):
    mae = sm.mean_absolute_error(y,pred_y)
    mse = sm.mean_squared_error(y,pred_y)
    mde = sm.median_absolute_error(y,pred_y)
    evs = sm.explained_variance_score(y, pred_y)
    r2 = sm.r2_score(y, pred_y)
    print("mae:{}%,mse:{}%,mde：{}%,evs:{}%,r2:{}%".format(mae,mse,mde,evs,r2))

def make_data(encoders):
    data = [
        ["Tuesday","13:35","San Francisco","yes"]
    ]
    data = np.array(data).T
    x = []
    for row in range(len(data)):
        encoder = encoders[row]
        x.append(encoder.transform(data[row]))
    x = np.array(x).T
    return x

def main():
    fileName = "./data/traffic.txt"
    x, y , encoders = read_data(fileName)
    train_x,test_x,train_y , test_y   = ms.train_test_split(x,y,test_size=0.25,random_state=5)
    model = train_model(train_x,train_y)
    pred_test_y = pred_model(model,test_x)
    eval_model(test_y,pred_test_y)

    new_x =  make_data(encoders)
    new_pred_y = pred_model(model,new_x)
    print("new_pred_y:", new_pred_y)
    print(encoders[-1].inverse_transform(new_pred_y))

if __name__ == '__main__':
    main()
