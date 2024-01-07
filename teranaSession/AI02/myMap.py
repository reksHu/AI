
def f1(x):
    return x+3

def myMap():
    b = [1,2,3,4,5,6]
    e = map(f1,b)
    for r in e:print(r,end='  ')

def f2(x,y):
    print("x={},y={}".format(x,y))
    return x+y

def f3(x,sent):
    print(x,sent.count('learning'))
    return x+sent.count('learning')

def f4(x,y,z):
    return x+y+z

def myReduce():
    import functools
    d = [x for x in range(1,7)]
    # print(d)
    # result = functools.reduce(f2,[1,2,3,4])
    r = functools.reduce(lambda x, y : f2(x,y),[1,2,3,4])
    print(r)

    # sentences = [
    #     'The Deep Learning textbook is a resource intended to help students and practitioners enter the field of machine learning in general and deep learning in particular. ']
    # r = functools.reduce(f3,sentences, 0)
    # print(r)

myReduce()
