#采用决策树预测 事件发生
import  numpy as np
import pandas as pd
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import sklearn.svm as svm
from income.DigitEncoder import MyDigitEncoder
def read_data(fileName):
    x = []
    with open(fileName,'r',encoding='utf-8') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            line_data = line[:-1].split(',')
            data.append(line_data)
        data = np.array(data).T
        data = np.delete(data,1,0)  #不需要具体日期类型，删除转置后第一行的日期类型，按照0轴(x轴)
    # print(data)
    encoders = []
    y = None
    for row in range(len(data)):
        if(data[row,0].isdigit()):
            encoder = MyDigitEncoder()
        else:
            encoder = sp.LabelEncoder()
        if(row<len(data)-1):
            x.append(encoder.fit_transform(data[row]))
        else:
            y = encoder.fit_transform(data[row])
        encoders.append(encoder)
    x = np.array(x).T
    return x, y ,encoders

def train_model(x, y ):
    model = svm.SVC(kernel='rbf',class_weight='balanced',probability=True) #使用SVM中的径向基内核训练， 数据采用 平衡数据训练
    model.fit(x,y)

    return  model

def make_data(encoders):
    data = [
        ["Monday","17:00:00","0","1"],
        ["Friday", "10:30:00", "12", "13"]
    ]
    data = np.array(data).T
    x = []
    for row in range(len(data)):
        encoder = encoders[row]
        x.append(encoder.transform(data[row]))
    x = np.array(x).T

    return x

def pred_model(model,test_x):
    pred_y = model.predict(test_x)
    return pred_y

#进行交叉验证
def eval_cv(model,x , y):
    pw = ms.cross_val_score(model,x,y,scoring="precision_weighted",cv = 2)
    rc = ms.cross_val_score(model, x, y, cv=2, scoring='recall_weighted')
    f1 = ms.cross_val_score(model, x, y, cv=2, scoring='f1_weighted')
    ac = ms.cross_val_score(model, x, y, cv=2, scoring='accuracy')
    print("{}% {}% {}% {}%".format(round(pw.mean() * 100, 2), round(rc.mean() * 10, 2),
                                   round(f1.mean() * 100, 2), round(ac.mean() * 100, 2)))


def eval_ac(y, pred_y):
    ac = ((y==pred_y).sum()/pred_y.size)
    print("Accuracy: {}%".format(round(ac * 100, 2)))

def get_prob(model,x):
    proba = model.predict_proba(x)
    return proba

def main():
    import pdb
    fileName = r"E:\python\study\terana_Learning\AI\data\event.txt"
    x , y, encoders =  read_data(fileName)
    train_x , test_x, train_y , test_y = ms.train_test_split(x, y ,test_size=0.25,random_state=5)
    model = train_model(train_x, train_y )
    eval_cv(model,x,y)

    pred_test_y = pred_model(model,test_x)
    eval_ac(test_y,pred_test_y)

    pred_x = make_data(encoders)
    pred_y = pred_model(model,pred_x)
    prob_y = get_prob(model,pred_x)
    print(encoders[-1].inverse_transform(pred_y))
    print(prob_y)
    # pred_test_y = pred_model(model,test_x)
    # proba_test_y = get_prob(model,test_x)
    # print(encoders[-1].inverse_transform(pred_test_y))
    # print(proba_test_y)
if __name__ == '__main__':
    main()