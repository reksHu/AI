import numpy as np


#
# k x       + b = y
# k col_min + b = 0
# k col_max + b = 1
# /           \   /   \   /   \
# | col_min 1 | X | k | = | 0 |
# | col_max 1 |   | b |   | 1 |
# \           /   \   /   \   /
#       a           x       b
#
def mmx_scale(mmx_sample,xmin, xmax):
    col_count = mmx_sample.shape[1]
    for col in range(col_count):
        col_sample = mmx_sample[:,col]
        col_min = col_sample.min()
        col_max = col_sample.max()
        A = np.array([[col_min,1],[col_max,1]])
        b_array = np.array([xmin,xmax])
        k, b = np.linalg.lstsq(A,b_array)[0]
        print(k, b)
        col_sample *=k
        col_sample +=b
    print(mmx_samples)

def mmxAPI(mmx_sample,xmin,xmax):
    import sklearn.preprocessing as sp
    mmx = sp.MinMaxScaler(feature_range=(xmin,xmax))
    mmx_sample =  mmx.fit_transform(mmx_sample)
    print(mmx_sample)

mmx_samples = np.array([
    [3, -1.5,    2, -5.4],
    [0,    4, -0.3,  2.1],
    [1,  3.3, -1.9, -4.3]])

# mmx_scale(mmx_samples,0,1)
mmxAPI(mmx_samples,0,1)