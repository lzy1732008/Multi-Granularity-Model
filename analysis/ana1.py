#分析每个法条相关的事实都都是属于哪些主题，统计平均的主题权重
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import jieba.posseg as pos
import jieba
import codecs
import random
import math

datapath = '../resource/train-init.txt'
ft = '中华人民共和国刑法(2015)第六十一条'
modelpath = '../result/lda/fact-5.model'
corpuspath = '../resource/ana/facts_corpus-cx.txt'

stopwords = codecs.open('../resource/stopwords.txt','r',encoding='utf8').readlines()
stopwords = [ w.strip() for w in stopwords]




def getSampleOfLaw(datapath,input_ft):
    lines = open(datapath,'r',encoding='utf-8').read().split('\n')
    samples = []
    for line in lines:
        line =line.strip()

        if line != "":
            terms = line.split('|')
            ft = terms[2]
            if str(ft).startswith(input_ft) and terms[-1] == '1':samples.append(terms[1])
    print(len(samples))
    index = random.sample([i for i in range(len(samples))],min(len(samples),100))

    samples = [samples[x] for x in index]
    return samples

def countTopicStatistic(corpuspath,samples,modelpath):
    lda = LdaModel.load(modelpath)
    train = []
    fp = codecs.open(corpuspath, 'r', encoding='utf8')
    for line in fp:
        line = line.strip()
        if line == '':continue
        line = line.split()
        train.append([w for w in line])

    dictionary = corpora.Dictionary(train)
    topics = []
    for sample in samples:
        words = pos.cut(sample) # 新文档进行分词
        input_words = []
        for w in words:
            if w.flag in ['n','v','a'] and w.word not in stopwords:
                input_words.append(w.word)
        doc_bow = dictionary.doc2bow(input_words)  # 文档转换成bow
        doc_lda = lda[doc_bow]  # 得到新文档的主题分布
        array = [0,0,0,0,0]
        for t in doc_lda:
            array[t[0]] = t[1]
        topics.append(array)
    print("平均主体分布")
    topics = np.array(topics)
    topics = np.mean(topics,axis=0)
    print(topics)


samples = getSampleOfLaw(datapath,ft)
countTopicStatistic(corpuspath,samples,modelpath=modelpath)
