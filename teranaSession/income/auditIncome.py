from DigitEncoder import MyDigitEncoder
# from .DigitEncoder import MyDigitEncoder
import numpy as np
import sklearn.naive_bayes as sh
import sklearn.model_selection as ms
import sklearn.preprocessing as sp
import sklearn.metrics as sm
import pdb

def read_data(fileName):
    x, y = [], []
    data,encoders = [], []
    num_less,more_less = 0,0 #分别记录高收入和收入 数据条数
    max_num = 7500 # 记录高收入和收入 数据条数分别最大7500条记录，用于数据集训练
    with open(fileName,'r') as f:
        lines = f.readlines()
        for line in lines:
            if('?' not in line):
                data_line = [d.strip() for d in line[:-1].split(",")]
                if(data_line[-1] =="<=50K" and num_less <=max_num):
                    data.append(data_line)
                    num_less +=1
                elif(data_line[-1]==">50K" and more_less<=max_num):
                    data.append(data_line)
                    max_num +=1
                if(more_less>max_num and num_less>max_num):
                    break
        data = np.array(data).T
        for row in range(len(data)):
            # pdb.set_trace()
            if(data[row,0].isdigit()):
                encoder = MyDigitEncoder()
            else:
                encoder = sp.LabelEncoder()
            if(row < len(data)-1):
                x.append(encoder.fit_transform(data[row]))
            else:
                y = encoder.fit_transform(data[row])
            encoders.append(encoder)
    x = np.array(x).T
    print(len(data),x.shape)
    # y = np.array(y)
    return x,y,encoders

def train_model(x,y):
    model = nb.GaussianNB()
    model.fit(x,y)
    return model

def predict_mode(model,x):
    pred_y = model.predict(x)
    return pred_y

def prepare_test_data():
    source_data = [
        ["39, Private, 77516, 9th, 5, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, Mexico"],
        ["56, Private, 284582, Masters, 14, Married-civ-spouse, Exec-managerial, Wife, White, Black, 0, 0, 40, United-States"],
        ["52, State-gov, 209642, HS-grad, 9, Married-civ-spouse, Exec-managerial, Husband, White, Male, 555, 0, 45, United-States"],
        ["23, Private, 122272, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Female, 0, 0, 30, Cuba"],
        ["25, Local-gov, 176756, HS-grad, 9, Never-married, Farming-fishing, Own-child, White, Male, 5178, 0, 35, South"],
        ["25, Self-emp-not-inc, 176756, HS-grad, 9, Separated, Farming-fishing, Own-child, White, Male, 0, 0, 35, Honduras"]
    ]
    data = []
    test_x = []
    for row in range(len(source_data)):
        line_data = [d.strip() for d in source_data[row].split(',')]
        data.append(line_data)
    data =  np.array(data).T
    for row in range(len(data)):
        if(data[row,0].isdigit()):
            encoder = MyDigitEncoder()
        else:
            encoder = sp.LabelEncoder()
        encoder.fit_transform(data[row])
        test_x.append(encoder)
    test_x =  np.array(test_x).T
    return test_x

#进行交叉验证
def eval_cv(model, x, y):
    pc = ms.cross_val_score(model,x,y,cv = 2,scoring='precision_weighted')
    rc = ms.cross_val_score(model, x, y, cv = 2, scoring='recall_weighted')
    f1 = ms.cross_val_score(model, x, y, cv = 2 ,scoring= 'f1_weighted')
    ac = ms.cross_val_score(model, x, y, cv = 2, scoring= 'accuracy')
    print("{}% {}% {}% {}%".format(round(pc.mean()*100,2) ,round(rc.mean()*10,2),
                                   round(f1.mean()*100,2), round(ac.mean()*100,2) ))

#检验模型精度
def eval_av(y,pred_y):
    ac = (y==pred_y).sum()/pred_y.size
    print("Accuracy {}%:".format(round(ac,2)))

#打印出分类器测试报告,里面包括了精度数据
def eval_cr(y,pred_y):
    cr = sm.classification_report(y,pred_y)
    print(cr)
def main():
    fileName = r"E:\python\study\terana_Learning\AI\data\adult.txt"
    x, y, encoders = read_data(fileName)
    print(y[:10],type(y))
    x_train, x_test, y_train, y_test = ms.train_test_split(x,y,test_size=0.25,random_state=5)
    model = train_model(x_train,y_train)
    eval_cv(model, x_train, y_train)

    pred_y = predict_mode(model,x_test)
    eval_av(y_test,pred_y)



main()
