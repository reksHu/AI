import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import matplotlib.pyplot as plt
import  numpy as np
import sklearn.ensemble as se

def read_data():
    hoursing = sd.load_boston()
    fn = hoursing.feature_names
    x, y = su.shuffle(hoursing.data,hoursing.target,random_state=7)

    return fn, x, y

#绘制决策树模型
def init_model_dt():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.subplot(211)
    plt.title('Decision Tree Regression',fontsize=16)
    plt.xlabel('Feature',fontsize = 12)
    plt.ylabel('Importance',fontsize=12)
    plt.tick_params(which='both',top =True, right = True, labelright =True , labelsize = 10)
    plt.grid(axis='y',linestyle=":")

def init_model_ab():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.show(212)
    plt.title('Ada Boost Decision Tree Regression',fontsize = 16)
    plt.xlabel('Feature',fontsize = 12)
    plt.ylabel('Importance', fontsize = 12)
    plt.tick_params(which ='both',top=True,right = True,labelright=True,labelsize = 10)
    plt.grid(axis='y',linestyle=':')

#fi_dt: 模型重要性
def draw_model_dt(fn,fi_dt):
    fi_dt = (fi_dt*100)/fi_dt.max()
    sorted_indices = np.flipud(fi_dt.argsort())
    pos = np.arange(sorted_indices.size)
    plt.bar(pos,fi_dt[sorted_indices],align = 'center',facecolor='deepskyblue',
            edgecolor = 'steelblue', label='Desicion tree',alpha=0.5)
    plt.xticks(pos,fn[sorted_indices])
    plt.legend()

def draw_model_ab(fn, fi_ab):
    fi_ab = (fi_ab*100)/fi_ab.max()
    sorted_indices = np.flipud(fi_ab.argsort())
    pos = np.arange(sorted_indices.size)
    plt.bar(pos,fi_ab[sorted_indices],align ='center',facecolor = 'lightcoral',edgecolor='indianred',label = 'Ada Boost Decision Tree')
    plt.legend()

def train_model_dt(x, y):
    model_dt = st.DecisionTreeRegressor(max_depth=4)
    model_dt.fit(x,y)
    return model_dt

def train_model_ab(x, y ):
    model_ab = se.AdaBoostRegressor(st.DecisionTreeRegressor(max_depth=4), n_estimators=400,random_state=7)
    model_ab.fit(x, y)
    return model_ab

def show_chart():
    plt.tight_layout()
    plt.show()

def main():
    fn, x, y = read_data()
    train_size = int(len(x)*0.8)
    train_x = x[:train_size]
    train_y = y[:train_size]
    model_dt = train_model_dt(train_x,train_y)
    model_ab = train_model_ab(train_x,train_y)
    fi_dt = model_dt.feature_importances_
    fi_ab = model_ab.feature_importances_
    init_model_dt()
    draw_model_dt(fn, fi_dt)
    init_model_ab()
    draw_model_ab(fn,fi_ab)
    show_chart()
main()
