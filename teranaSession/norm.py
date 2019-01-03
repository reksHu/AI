#归一化
import  numpy as np
import sklearn.preprocessing as sp
def normalization(raw_sample):
    row_count = raw_samples.shape[0]
    for row in range(row_count):
        row_sample = raw_sample[row]
        abs_sample = abs(row_sample)
        row_sum = abs_sample.sum()
        # print("row_sample sum:",row_sum)
        row_sample /= row_sum
        # print("row_sample:",row_sample)
        # print("result sum:",row_sample.sum())
        # print(abs(row_sample))
    return raw_sample

def normalizationAPI(raw_sample):
    sp.normalize(raw_sample,norm=11)
#使用绝对值归一化，测试
def test():
    nor_sample = raw_samples.copy()
    rows = nor_sample.shape[0]
    for r in range(rows):
        row_sample = nor_sample[r]
        row_abs =abs(row_sample)
        row_abs_sum = row_abs.sum()
        row_sample /= row_abs_sum
    print(nor_sample)
    return nor_sample


raw_samples = np.array([
    [3, -1.5, 2, -5.4],
    [0, 4, -0.3, 2.1],
    [1, 3.3, -1.9, -4.3]])
print("source data:",raw_samples)
raw_samples =  normalization(raw_samples)
print("rsult:",raw_samples)

print(abs(raw_samples[0]).sum())

# nor_sample = test()
# print(nor_sample[0].sum())