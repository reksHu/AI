import numpy as np
def binarize(raw_sample,thredhold):
    raw_sample[raw_sample<=thredhold] = 0
    raw_sample[raw_sample>thredhold] = 1
    return raw_sample

def binarizeAPI(raw_sample,thredholdVal):
    import sklearn.preprocessing as sp
    bin = sp.Binarizer(threshold = thredholdVal)
    result = bin.transform(raw_sample)
    print(result)


raw_samples = np.array([
    [3, -1.5,    2, -5.4],
    [0,    4, -0.3,  2.1],
    [1,  3.3, -1.9, -4.3]])

# res = binarize(raw_samples,1.4)
# print(res)
# print(raw_samples)  #原始数列已经改变
binarizeAPI(raw_samples,1.4)
print(raw_samples)

