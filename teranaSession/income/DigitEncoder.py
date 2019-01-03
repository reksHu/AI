
class MyDigitEncoder():
    # 将数字型字符串编码为int
    def fit_transform(self, y):
        return y.astype(int)

    def transform(self, y):
        return y.astype(int)

    # 将数字类型转换为字符串类型
    def inverse_transform(self, y):
        return y.astype(str)
