from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.Text8Corpus(u"E:\\python\\tools\\text8")
model = word2vec.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5

model.save("text8.model")

