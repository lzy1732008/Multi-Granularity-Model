import os
from util.ws_fun import getQWChildContent
import jieba.analyse as ana
import jieba
import re
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def CoOccur(wsDir):
    stpwords = open('../resource/stopwords.txt','r',encoding='utf-8').readlines()

    vocab = {}# "word1":{"word1":1,"word2":2}
    files = os.listdir(wsDir)
    corpus = []
    lines_keywords = []
    for file in files[:500]:
        if not str(file).endswith('.xml'): continue
        wspath = os.path.join(wsDir,file)
        content = getQWChildContent(wspath,'CPFXGC').split('。')
        for c in content:
            keywords = jieba.analyse.textrank(c,topK=5,allowPOS=['n','v','a'])
            for word in keywords:
                if word not in vocab.keys():
                    vocab[word] = {}

                for i in range(len(keywords)):
                    if keywords[i] == word:continue
                    if keywords[i] not in vocab[word].keys():
                        vocab[word][keywords[i]] = 0
                    vocab[word][keywords[i]] += 1

    for k,v in vocab.items():
        keywords = sorted(v, key=lambda x:x[1], reverse=True)
        if len(keywords) > 5:
            keywords = keywords[:5]
        print('word:{0},reltaed word:{1}'.format(k,' '.join(keywords)))







        # corpus += [' '.join(jieba.lcut(x)) for x in content]

    # vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    # transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    # tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # words = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    # weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    # line_keywords = []
    # for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    #     word_dic = {}
    #     for j in range(len(words)):
    #         if words[j] in corpus[i]:
    #             word_dic[words[j]] = weight[i][j]
    #
    #     if len(word_dic) >5 : word_dic = sorted(word_dic,key=lambda x:x[1],reverse=True)[:5]
    #     line_keywords.append(word_dic)
    #     print('{0}:{1}'.format(corpus[i], '/'.join(word_dic)))








#
# wsDir = '/Users/wenny/nju/task/LawDocumentAna/2014filled/2014'
# CoOccur(wsDir)
#

