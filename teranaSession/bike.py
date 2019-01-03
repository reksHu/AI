
import numpy as np
import pandas as pd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import csv
def read_data(filename,fa,fb):
    x, y = [], []
    with open(filename,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            x.append(row[fa:fb])
            y.append(row[-1])
        fn = np.array(x[0])
        x = np.array(x[1:],dtype=np.float)
        y = np.array(y[1:],dtype=np.float)
        x, y = su.shuffle(x, y, random_state = 7)
    return fn, x, y

#min_samples_split最小两个样本开始样本分裂
def train_model(x, y ):
    model_dt = se.RandomForestRegressor(max_depth=20,n_estimators=50,min_samples_split=2)
    model_dt.fit(x, y)
    return model_dt

def pred_model(model,x):
    y = model.predict(x)
    return y

def eval_model(y,pred_y):
    mae =  sm.mean_absolute_error(y,pred_y)
    mse = sm.mean_squared_error(y,pred_y)
    mde = sm.median_absolute_error(y, pred_y)
    evs = sm.explained_variance_score(y, pred_y)
    r2 = sm.r2_score(y,pred_y)
    print("mae:", mae, "mse:", mse, 'mde:', mde, 'evs:', evs, 'r2:', r2)


def init_model_day():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.subplot(211)
    plt.title('Random Rorest Regresion By Day',fontsize = 16)
    plt.xlabel('Feature',fontsize = 12)
    plt.ylabel('Importance',fontsize = 12)
    plt.tick_params(which ='both',top =True,right = True, labelright=True,labelsize = 10)
    plt.grid(axis='y',linestyle = ':')

def draw_model_day(fn, fi_day):
    fi_day = (fi_day*100) / fi_day.max()
    sorted_incidencs = np.flipud(fi_day.argsort())
    pos = np.arange(sorted_incidencs.size)
    plt.bar(pos,fi_day[sorted_incidencs],align = 'center',facecolor = 'deepskyblue',
            edgecolor = 'steelblue',label = 'Decision Tree by Day')
    plt.xticks(pos,fn[sorted_incidencs])
    plt.legend()
    plt.show()


fn, x_day, y_day = read_data('./data/bike_day.csv',2,13)
train_day_size = int(len(x_day)*0.8)
train_x_day = x_day[:train_day_size]
train_y_day = y_day[:train_day_size]
test_x_day = x_day[train_day_size:]
test_y_day = y_day[train_day_size:]
model_day = train_model(train_x_day,train_y_day)
fi_day = model_day.feature_importances_
pred_y_day = pred_model(model_day,test_x_day)
print(pred_y_day)
eval_model(test_y_day,pred_y_day)
init_model_day()
draw_model_day(fn,fi_day)




