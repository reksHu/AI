import numpy as np
import json
import pandas as pd

def read_data(fileName):
    with open(fileName,'r') as f:
        ratings = json.loads((f.read()))

    return ratings

#皮尔逊距离计算
def calc_ps(ratings,user1,user2):
    movies = set()
    for movie in ratings[user1]:
        if movie in ratings[user2]:
            movies.add(movie)
    n = len(movies)
    if(n == 0):
        return 0
    x = np.array([ratings[user1][movie] for movie in movies])
    y = np.array([ratings[user2][movie] for movie in movies])

    sx = x.sum()
    sy = y.sum()
    xx = np.square(x).sum()
    yy = (y**2).sum()
    xy = (x * y).sum()
    sxx = xx - sx ** 2 /n
    syy = yy - sy ** 2/n
    sxy = xy - sx * sy /n
    if(sxx * syy == 0):
        return 0
    pearson_socre = sxy / np.sqrt(sxx * syy)  # np.sqrt(sxx * syy)是为了去掉自己本身的相关性
    return pearson_socre

#生成皮尔逊距离 用户和得分举证
def eval_ps(ratings):
    users, psmat = list(ratings.keys()), []
    for user1 in users:
        psrow = []
        for user2 in users:
            psrow.append(calc_ps(ratings,user1,user2))
        psmat.append(psrow)

    users = np.array(users)
    psmat = np.array(psmat)
    return users,psmat

def find_similar(users,psmat,user1, n_similar):
    # index = users.index(user1) #查找index , users 需要为list
    user_index = np.arange(len(users))[users==user1][0]
    sorted_index = psmat[user_index].argsort()[::-1]# 取出相似分值按照降序排列
    sorted_index = sorted_index[sorted_index!=user_index][:n_similar]
    user_socre = psmat[user_index][sorted_index]
    user_similar = users[sorted_index]
    return user_socre, user_similar

def main():
    fileName = r"E:\python\study\terana_Learning\AI\data\ratings.json"
    ratings = read_data(fileName)
    # print(ratings)
    users, psmat = eval_ps(ratings)
    user1 = "John Carson"
    n_similar = 3 #指定查询多少个相似度的用户
    user_socre, user_similar = find_similar(users,psmat,user1,n_similar)
    data = pd.DataFrame({"Users":user_similar,"Score":user_socre})
    print(data)
    print(ratings)
    # users = np.array(users)
    # result = np.arange(len(users))[users == user1]
    # print(result)

main()
