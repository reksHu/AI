import numpy as np
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.ensemble as se  #自适应增强 , ada boost
import sklearn.metrics as sm

def read_data():
    hoursing = sd.load_boston()
    x, y = su.shuffle(hoursing.data, hoursing.target,random_state = 7) #对两个数列进行洗牌，洗牌后data与target原来对应项也对应，按照行洗牌，random states 描述随机度
    return  x, y

def train_model_dt(x, y):
    mode_dt = st.DecisionTreeRegressor(max_depth=4) #定义最大深度为4的回归决策树模型
    mode_dt.fit(x, y )
    return  mode_dt

#带有自适应增强的决策树
# n_estimators 评估器, random_state随机度
def train_mode_ad(x, y ):
    model_ab = se.AdaBoostRegressor(st.DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state= 7 )
    model_ab.fit(x, y )
    return  model_ab

def eval_model(y, pred_y):
    mae = sm.mean_absolute_error(y,pred_y) #所有数据点的误差绝对值的平均数
    mse = sm.mean_squared_error(y, pred_y) #所有数据点的误差平方值得平均数
    mde = sm.median_absolute_error(y, pred_y) # 数据集中位数误差的绝对值中位数,有利于排除某些奇异值(Outlier干扰)
    evs = sm.explained_variance_score(y, pred_y)
    r2 = sm.r2_score(y, pred_y)
    print("mae:",mae,"mse:",mse,'mde:',mde,'evs:',evs,'r2:',r2)

def pred_model(model, x):
    y = model.predict(x)
    return  y

def main():
    x, y = read_data()
    train_x = x[:int(len(x)*0.8)]
    test_x = x[int(len(x)*0.8):]
    train_y = y[:int(len(y)*0.8)]
    test_y = y[int(len(y)*0.8):]


    model = train_model_dt(train_x,train_y)
    pred_train_y = pred_model(model,train_x)
    eval_model(train_y,pred_train_y)

    model_ad = train_mode_ad(train_x,train_y)
    pred_ad_train_y = pred_model(model_ad,train_x)
    eval_model(train_y,pred_ad_train_y)

x, y = read_data()
print(type(x),type(y))
