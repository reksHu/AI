import numpy as np
def test1():
    raw_samples = np.array([
        [3, -1.5,    2, -5.4],
        [0,    4, -0.3,  2.1],
        [1,  3.3, -1.9, -4.3]])
    # x = np.array([[i for i in range(11,16)],[x for x in range(1,6)]])
    # print(x)
    # print(x.mean())
    std_sample = raw_samples.copy()

    colNum = std_sample.shape[1]
    # print(raw_samples[:,1])
    col_sample = std_sample[:,1]

    print("col_sample:",col_sample)
    col_mean = col_sample.mean()
    col_std = col_sample.std()
    print("std:",col_std)
    print("col_mean:",col_mean)
    col_sample -=col_mean
    print("移除平均值后 col_sample:",col_sample)
    col_sample /=col_std
    print("std_sample",std_sample)
    print("mean,",std_sample[:,1].std())

def test2():
    import math
    sample = np.array([x for x in range(1,6)],dtype= np.float16)
    print(sample)
    samp_mean = sample.mean()
    samp_std = sample.std()
    print(samp_mean , samp_std)

    print("*"*30)
    sample -= samp_mean
    print("removed mean:",sample)
    sample /= samp_std
    print("std 1: ", sample.std())

#sciktlearn 对标准差缩放
def scale():
    import sklearn.preprocessing as sp
    sample = np.array([x for x in range(1, 6)], dtype=np.float16)
    sample = sp.scale(sample)
    print(sample, sample.mean(), sample.std())

scale()

#
# for col in range(colNum):
#     r =  raw_samples[:,col]
#     print(r)

