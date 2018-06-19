from gensim.models import word2vec
import  logging
def clearnData():
    fileName ='sp.txt'
    with open(fileName,'r',encoding='utf-8') as f:
        lines =  f.readlines()
        for line in lines:
            line = line.replace('.',' ').replace(',',' ')
            print(line.lower())
            cleanPath = "spCleanData.txt"
            with open(cleanPath,'a',encoding='utf-8') as txt_f:
                txt_f.write(line.lower())


def trainSpData():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    cleanPath = "spCleanData.txt"
    sentences = word2vec.Text8Corpus(cleanPath)
    model = word2vec.Word2Vec(sentences, size=200)

    y1 = model.wv.similarity("sharepoint", "server")
    print(y1)

trainSpData()
