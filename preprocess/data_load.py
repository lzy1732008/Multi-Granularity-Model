# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import numpy as np
from preprocess import preprocess as pre
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


