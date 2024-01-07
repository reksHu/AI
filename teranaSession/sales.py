import numpy as np
import csv
import sklearn.cluster as sc
import sklearn.metrics as sm

def read_data(fileName):
    with open(fileName,'r') as f:
        csv_data = csv.reader(f)
        x = []
        for data in csv_data:
            x.append(data[2:])
        features =np.array(x[0]) #获取产品名称
        x = np.array(x[1:],dtype = np.float)
        print(x[:10],features)
        return  x, features

def train_model(x): #采用均值漂移训练数据
    bw = sc.estimate_bandwidth(x,n_samples=len(x),quantile=0.8)
    model = sc.MeanShift(bandwidth=bw,bin_seeding=True)
    model.fit(x)
    score = sm.silhouette_score(x,sample_size= len(x), labels  = model.labels_)
    print("Score:",score)
    return model

def pred_model(model,x):
    pred_y = model.predict(x)
    return pred_y

def main():
    fileName = r"E:\python\study\terana_Learning\AI\data\sales.csv"
    x, features = read_data(fileName)
    model = train_model(x)
    pred_y = pred_model(model,x)
    print(pred_y)

main()
