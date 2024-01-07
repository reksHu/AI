
import functools

#实现 (a+3)*6 - 9

def f1(x):
    print("f1 x = ",x)
    return x + 3

def f2(x):
    print("f2 x = ", x)
    return x * 6

def f3(x):
    print("f3 x = ", x)
    return x - 9

funs = [f1,f2,f3]
a = 1
# r = functools.reduce(lambda fa,fb:lambda x : fa(fb(x)) , [f3,f2,f1])(a)

r1 = functools.reduce(lambda fa,fb:lambda x:fb(fa(x)),[f1,f2,f3])(a)
print(r1)


import sklearn.pipeline as spip
import sklearn.preprocessing as sp
import sklearn.linear_model as sl
model = spip.make_pipeline(sp.PolynomialFeatures(degree= 10) , sl.LinearRegression())
model.fit()

