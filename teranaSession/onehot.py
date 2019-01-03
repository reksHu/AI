import numpy as np

raw_samples = np.array([
    [0, 0, 3],
    [1, 1, 0],
    [0, 2, 1],
    [1, 0, 2]])



def onehot_code(sample):
    onehot_sample = sample.T
    print(onehot_sample)
    codes = []
    for row in sample.T:
        code_dict = {}
        for col in row:
            code_dict[col]=None
        codes.append(code_dict)
    print("codes", codes)
    for code_table in codes:
        tableLen = len(code_table)
        for index,value in enumerate(sorted(code_table.keys())):
            code_table[value] = np.zeros(tableLen,dtype=np.int16)
            code_table[index][value] = 1
    print(codes)

    for index,key in enumerate(sample):
        print(index,key)
    onehot = None

    return onehot

onehot_code(raw_samples)

