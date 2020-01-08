import codecs
import jieba
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary

def train(corpuspath,modelpath):
    train = []
    # stopwords = codecs.open('stopWords/1893（utf8）.txt','r',encoding='utf8').readlines()
    # stopwords = [ w.strip() for w in stopwords ]
    fp = codecs.open(corpuspath, 'r', encoding='utf8')
    for line in fp:
        line = line.strip()
        if line == '':continue
        line = line.split()
        train.append([w for w in line])

    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=20)
    lda.save(modelpath)

    topic_words = open('../result/lda/fact_lad_print-10.txt','w',encoding='utf-8')
    print_str = ''
    for topic in lda.print_topics(num_words=100):
        termNumber = topic[0]
        listOfTerms = topic[1].split('+')
        # for term in listOfTerms:
        #     listItems = term.split('*')
        #     # print(listItems)
        #     print('  ', listItems[1], '(', listItems[0], ')', sep='')
        print_str += topic[1] + '\n'
    topic_words.write(print_str)
    topic_words.close()


    # predict
    # lda = LdaModel.load('result/中华人民共和国刑法(2015)第一百三十三条_lda.model')
    # test_doc = list(jieba.cut('经兰州市公安局交通警察支队红古大队认定，陈天骅承担本次事故主要责任'))  # 新文档进行分词
    # doc_bow = dictionary.doc2bow(test_doc)  # 文档转换成bow
    # doc_lda = lda[doc_bow]  # 得到新文档的主题分布
    # # 输出新文档的主题分布
    # print(doc_lda)
    # for topic in doc_lda:
    #     print("%s\t%f\n" % (lda.print_topic(topic[0]), topic[1]))


import os
corpus_path = '../resource/facts_corpus.txt'
# model_path = '../result/lda/fact_lda-10.model'
# train(corpus_path,model_path)

#
# def predictTopic():
#     corpus_path = '../resource/facts_corpus.txt'
#     train = []
#     fp = codecs.open(corpus_path, 'r', encoding='utf8')
#     for line in fp:
#         line = line.strip()
#         if line == '': continue
#         line = line.split()
#         train.append([w for w in line])
#
#     dictionary = corpora.Dictionary(train)
#     lda = LdaModel.load('../result/lda/fact_lda-10.model')
#     for line in train[:100]:
#         test_doc = line
#         doc_bow = dictionary.doc2bow(test_doc)  # 文档转换成bow
#         doc_lda = lda[doc_bow]  # 得到新文档的主题分布
#         # 输出新文档的主题分布
#         print(doc_lda)
#         print('==================================================')
#         print(line)
#         for topic in doc_lda:
#             print("%s\t%f\n" % (lda.print_topic(topic[0]), topic[1]))


#
# predictTopic()