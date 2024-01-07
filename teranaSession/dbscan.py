import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as plt
import sklearn.metrics as sm

def read_data(file_name):
    data, x = [], []
    with open(file_name,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line_data = line[:-1].split(',')
            data = [float(d) for d in line_data]
            x.append(data)
    return np.array(x)

def train_model(x):
    epsilons = np.arange(0.3,1.3,0.1) # 等价于 epsilons = np.linspace(0.3,1.2,10)
    scores = []
    models = []
    for eps in epsilons:
        #eps: 两个样本间的最大距离. min_samples 最少5个样本组成一个集群
        model = sc.DBSCAN(eps=eps,min_samples=5).fit(x)
        models.append(model)
        score = sm.silhouette_score(x,model.labels_,sample_size=len(x),metric='euclidean')
        scores.append(score)
    scores = np.array(scores)
    best_index =  scores.argmax()
    best_score = scores[best_index]
    best_model = models[best_index]
    print("Score:",best_score,epsilons[best_index])

    return best_model

def pred_model(model,x):
    y = model.fit_predict(x)
    core_mask = np.zeros(len(x),dtype=bool)
    core_mask[model.core_sample_indices_] = True #获取核心掩码
    print(core_mask)
    off_mask = model.labels_ ==-1  #获取带外掩码
    periphery_mask = ~(core_mask | off_mask) #获取带外掩码，核心掩码和带外掩码之外的都是边缘掩码
    return y,core_mask,off_mask, periphery_mask

def init_chart():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.title('DBScan Cluster',fontsize = 20)
    plt.xlabel('x',fontsize = 14)
    plt.ylabel('y',fontsize = 14)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator()) #主坐标，刻度为1
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5)) #辅坐标，刻度为0.5
    ax.yaxis.set_major_locator(plt.MultipleLocator())
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))

    plt.tick_params(which='both',top = True, right = True, labelright = True, labelsize = 10)
    plt.grid(linestyle = ":")

def draw_data(x,y,core_mask,off_mask, periphery_mask):
    lables = set(y)
    cs = plt.get_cmap('brg',len(lables))(range(len(lables)))
    print(x[core_mask][:,0])
    plt.scatter(x[core_mask][:,0],x[core_mask][:,1],c=cs[y[core_mask]], s=80,label = 'Core')
    plt.scatter(x[off_mask][:,0],x[off_mask][:,1],c = cs[y[off_mask]], marker= 'x',s = 80,label = 'Offset')
    plt.scatter(x[periphery_mask][:,0],x[periphery_mask][:,1], facecolor ='none' ,edgecolors=cs[y[periphery_mask]],s = 80, label = "Periphery")
    plt.legend()
def show_chart():
    # mng = plt.get_current_fig_manager()
    # print(dir(mng.window.state('zoomed')))
    plt.show()


def main():
    fileName =r"E:\python\study\terana_Learning\AI\data\perf.txt"
    x =  read_data(fileName)

    model = train_model(x)

    pred_y, core_mask ,off_mask , periphery_mask= pred_model(model,x)

    init_chart()
    draw_data(x,pred_y, core_mask,off_mask, periphery_mask)

    show_chart()
    return 0



if __name__ == '__main__':
    main()