from gensim.models import word2vec
import logging
#https://blog.csdn.net/churximi/article/details/51472203
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# sentences = word2vec.Text8Corpus(u"E:\\python\\tools\\text8")
# model = word2vec.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5
model = word2vec.Word2Vec.load("text8.model")


# y1 = model.similarity("woman", "man")
# print(u"woman和man的相似度为：", y1)
# print("--------\n")

# 计算某个词的相关词列表
# y2 = model.most_similar("good", topn=20)  # 20个最相关的
# y2 = model.wv.most_similar("how are you".split(),topn=20)
# print(u"和good最相关的词有：\n")
# for item in y2:
#     print(item[0], item[1])
# print("--------\n")

# example = "how are "
# a, b = example.split()
# predicted = model.wv.most_similar_cosmul(positive=['baghdad', 'england'], negative=['london'])
# # predicted = model.wv.most_similar(positive=[a,b])
# print(predicted)
# predicted = model.wv.most_similar_cosmul(positive=['england','london'])
# print(predicted)

# result = model.score(["The fox jumped over a lazy dog".split()])
# print(result)

# more_examples = ["he his she", "big bigger bad", "going went being"]
# for example in more_examples:
#     a, b, x = example.split()
#     model.wv.most_similar()
#     predicted = model.most_similar([x, b], [a])[0][0]
#     print("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))
# print("--------\n")
#
# # 寻找不合群的词
# y4 = model.doesnt_match("breakfast cereal dinner lunch".split())
# print(u"不合群的词：", y4)
#
# print("--------\n")
#
# # 保存模型，以便重用
# model.save("text8.model")
# 对应的加载方式
# model_2 = word2vec.Word2Vec.load("text8.model")
#
# print("--------\n")
#
# # 以一种C语言可以解析的形式存储词向量
# model.save_word2vec_format("text8.model.bin", binary=True)
# 对应的加载方式
# model_3 = word2vec.Word2Vec.load_word2vec_format("text8.model.bin", binary=True)