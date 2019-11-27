# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import numpy as np
from preps import preprocess as pre
import tensorflow.contrib.keras as kr
import models.parameter as param

def get_batch_data_test(a_word,b_word,y, batch_size = 64):
    data_len = len(a_word)
    num_batch = int(data_len/batch_size)

    for i in range(num_batch):
        start_id = batch_size * i
        end_id = min(batch_size * (i + 1), data_len)
        yield a_word[start_id:end_id],\
              b_word[start_id:end_id],\
              y[start_id:end_id]


def get_batch_data(a_word,b_word,y,batch_size = 64):
    data_len = len(a_word)
    num_batch = int(data_len/batch_size)

    indices = np.random.permutation(np.arange(data_len))
    a_word_shuffle = a_word[indices]
    b_word_shuffle = b_word[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = batch_size * i
        end_id = min(batch_size * (i + 1), data_len)
        yield a_word_shuffle[start_id:end_id],\
              b_word_shuffle[start_id:end_id],\
              y_shuffle[start_id:end_id]


def data_load(trainPath, valPath, testPath,model):
    env = pre.setUp_inputs(trainPath=trainPath, valPath=valPath, testPath=testPath)
    train_data = env['train']
    test_data = env['test']
    val_data = env['val']
    train = processInitData(train_data,model)
    test = processInitData(test_data,model)
    val = processInitData(val_data,model)
    return train,test,val

def data_load_test(model):
    env = pre.setUp_inputs(trainPath=None, valPath=None,
                            testPath=param.BaseConfig.testPath)
    test_data = env['test']
    test = processInitData(test_data,model)
    return test


#process data from env
def processInitData(data,model):
    a_data_word = []
    b_data_word = []
    y = []

    for sample in data:
        assert len(sample) == 3, ValueError("the number of elemengs in this sample is {0}".format(len(sample)))
        input_a, input_b, target_y = sample[0],sample[1],int(sample[2])
        a_words = input_a['word_input']
        b_words = input_b['word_input']
        a_data_word.append(list(map(lambda x:pre.getVector(x), a_words)))
        b_data_word.append(list(map(lambda x:pre.getVector(x), b_words)))
        if target_y == 1:
            y.append([0,1])
        else:
            y.append([1,0])


    a_data_word = kr.preprocessing.sequence.pad_sequences(np.array(a_data_word), model.config.X_maxlen)
    b_data_word = kr.preprocessing.sequence.pad_sequences(np.array(b_data_word), model.config.Y_maxlen)
    return a_data_word,b_data_word, np.array(y)

import  re
def processInitDataLaw(data,model,wordEmbedding, vocabs):
    pattern = '，|。|；|：'
    regx = re.compile(pattern)
    inputX = []
    inputSplit = []
    y = []
    for index,sample in enumerate(data):
        input, target_y = sample[0], int(sample[1])
        array = regx.split(input)
        assert len(array) in [2,3,4], ValueError("Contain wrong number of sub items:{0} with {1} sub items".format(input,len(array)))
        if len(array) == 2: #单独的一个句子
            pcut = len(input) - 2
            hcut = 1
        elif len(array) == 3:
            assert input[0] == 'S' or input[-1] == 'E', ValueError("Wrong input text:"+input)
            if input[0] == 'S':
                pcut = len(array[0])
                hcut = 1
            elif input[-1] == 'E':
                pcut = len(input) - 2
                hcut = len(input) - len(array[-2]) - 2

        elif len(array) == 4:
            pcut = len(input) - len(array[-2]) - 2
            hcut = len(array[0])

        input_embed = pre.processLawText(input, wordEmbedding, vocabs)
        if len(input_embed) == 0 or input_embed is None: continue

        inputX.append(input_embed)

        pcut = min(model.config.max_len - 1,pcut)
        hcut = min(model.config.max_len - 1,hcut)

        inputSplit.append([pcut,hcut])
        if target_y == 0:
            y.append([1,0,0])
        elif target_y == 1:
            y.append([0,1,0])
        else:
            y.append([0,0,1])

    inputX = kr.preprocessing.sequence.pad_sequences(np.array(inputX), model.config.max_len)
    inputSplit = np.array(inputSplit)
    y = np.array(y)
    return [inputX,inputSplit,y]


import json
def dataLoadLaw(model, train=True, val=True, test=False):
    fr = open('resource/lawDataSet.json','r',encoding='utf-8')
    env = json.load(fr)
    trainSet, valSet, testSet = [], [], []
    if train:
        trainSet = env['trainSet']
    if val:
        valSet = env['valSet']
    if test:
        testSet = env['testSet']

    #load word embedding
    f_word = open(param.BaseConfig.w2vModel_ex, 'r', encoding='utf- 8')
    wordEmbedding = json.load(f_word)
    if '<UNK>' not in wordEmbedding.keys():
        wordEmbedding['<UNK>'] = '\t'.join(['0' for _ in range(param.BaseConfig.word_dimension)])

    wordVocab = wordEmbedding.keys()

    output = []
    #vectorization input
    if train:
       trainSet = processInitDataLaw(trainSet, model, wordEmbedding,wordVocab)
       output.append(trainSet)
    if val:
       valSet = processInitDataLaw(valSet, model, wordEmbedding,wordVocab)
       output.append(valSet)
    if test:
       testSet = processInitDataLaw(testSet, model, wordEmbedding,wordVocab)
       output.append(testSet)
    return output

def addIndexForLaw(dataSet):
    newDataSet = []
    for i in range(len(dataSet)):
        if i == 67:
            print("Error")
        temp = dataSet[i].tolist()
        temp.insert(0,i)
        newDataSet.append(np.array(temp))
    return np.array(newDataSet)

def checkNoneType(dataSet,wrongInfo,rightInfo):
    for i in range(len(dataSet)):
        assert dataSet[i] is not None, ValueError("wrong info:{0},index:{1},content:{2}".format(wrongInfo,i,dataSet[i]))
        for j in range(len(dataSet[i])):
            assert dataSet[i][j] is not None, ValueError("wrong info:{0},index1:{1},index2:{2},content:{3}".format(wrongInfo,i,j,dataSet[i]))
    # print(rightInfo)







