import numpy as np
import json


def read_data(fileName):
    with open(fileName,'r') as f:
        ratings = json.loads((f.read()))

    return ratings

#计算欧式距离
def calc_es(ratings,user1,user2):
    movies = set()
    for movie in ratings[user1]:
        if movie in ratings[user2]:
            movies.add(movie)
        if(len(movies)==0):
            return 0
    diffs = []
    for movie in movies:
        diffs.append(np.square(ratings[user1][movie] - ratings[user2][movie] ))
    diffs = np.array(diffs)
    euclidean_score = 1 / (1 + np.square(diffs.sum()))
    return euclidean_score

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

#生成欧式距离 用户和得分举证
def eval_es(ratings):
    users, esmat = list(ratings.keys()), []
    for user1 in users:
        esrow = []
        for user2 in users:
            esrow.append(calc_es(ratings,user1,user2))
        esmat.append(esrow)

    users = np.array(users)
    esmat = np.array(esmat)
    return users,esmat

def main():
    fileName = r"E:\python\study\terana_Learning\AI\data\ratings.json"
    ratings = read_data(fileName)
    users , psmat = eval_ps(ratings)
    print(users)
    print(psmat)

main()