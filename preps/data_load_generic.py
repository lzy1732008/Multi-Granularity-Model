from __future__ import print_function
import numpy as np
from preps import preprocess as pre
import tensorflow.contrib.keras as kr
import models.parameter as param

def get_batch_data_test(*data, batch_size=64):
    data_len = len(data[0])
    num_batch = int(data_len/batch_size)

    for i in range(num_batch):
        start_id = batch_size * i
        end_id = min(batch_size * (i + 1), data_len)
        yield [data[j][start_id:end_id] for j in range(len(data))]



def get_batch_data(*data, batch_size = 64):
    data_len = len(data[0])
    num_batch = int(data_len/batch_size)

    indices = np.random.permutation(np.arange(data_len))
    data_shuffle = []
    for i in range(len(data)):
        data_shuffle.append(data[i][indices])

    for i in range(num_batch):
        start_id = batch_size * i
        end_id = min(batch_size * (i + 1), data_len)
        yield [np.array(data[j][start_id:end_id]) for j in range(len(data_shuffle))]

import json

def data_load(trainPath, valPath, testPath,model,rfModel):
    env = pre.setUp_inputs_QHJ(trainPath=trainPath,valPath=valPath,testPath=testPath,rfModel=rfModel)
    train_data = env['train']
    test_data = env['test']
    val_data = env['val']
    train = []
    test = []
    val = []

    if trainPath:
       train = processInitData_Generic_OneHotQHJ(train_data,model)
    if valPath:
       val = processInitData_Generic_OneHotQHJ(val_data,model)
    if testPath:
       test = processInitData_Generic_OneHotQHJ(test_data,model)
    #
    #
    #
    with open('resource/dataSet50oneHotKS.json','w',encoding='utf-8') as fw:
        dataset = {}
        dataset['train'] = [train[0].tolist(), train[1].tolist(), train[2].tolist(), train[3].tolist(),train[4].tolist()]
        dataset['val'] = [val[0].tolist(), val[1].tolist(), val[2].tolist(), val[3].tolist(), val[4].tolist()]
        dataset['test'] = [test[0].tolist(), test[1].tolist(), test[2].tolist(), test[3].tolist(), test[4].tolist()]
        json.dump(dataset, fw)

    # with open('resource/dataSet50oneHotKS.json', 'r', encoding='utf-8') as fr:
    #     dataset = json.load(fr)
    #     train = dataset['train']
    #     val = dataset['val']
    #     test = dataset['test']
    #
    #     train = np.array(train[0]),np.array(train[1]),np.array(train[2]),np.array(train[3]), np.array(train[4])
    #     val = np.array(val[0]), np.array(val[1]), np.array(val[2]), np.array(val[3]), np.array(val[4])
    #     test = np.array(test[0]), np.array(test[1]), np.array(test[2]), np.array(test[3]), np.array(test[3])

    return train,val, test

def data_load_test(model,rfModel):
    env = pre.setUp_inputs_QHJ(trainPath=None, valPath=None,
                            testPath=param.BaseConfig.testPath,rfModel=rfModel)
    test_data = env['test']
    test = processInitData(test_data,model)
    return test

def processInitDataWithoutQHJ(data,model):
    a_data_word = []
    b_data_word = []
    y = []

    for sample in data:
        assert len(sample) == 3, ValueError("the number of elemengs in this sample is {0}".format(len(sample)))
        input_a, input_b,  target_y = sample[0],sample[1],int(sample[2])
        a_data_word.append(list(map(lambda x:pre.getVector(x), input_a)))
        b_data_word.append(list(map(lambda x:pre.getVector(x), input_b)))
        if target_y == 1:
            y.append([0,1])
        else:
            y.append([1,0])

    a_data_word = kr.preprocessing.sequence.pad_sequences(np.array(a_data_word), model.config.X_maxlen)
    b_data_word = kr.preprocessing.sequence.pad_sequences(np.array(b_data_word), model.config.Y_maxlen)
    return a_data_word,b_data_word, np.array(y)

#process data from env
def processInitData(data,model):
    a_data_word = []
    b_data_word = []
    c_data_word = []
    y = []

    for sample in data:
        assert len(sample) == 4, ValueError("the number of elemengs in this sample is {0}".format(len(sample)))
        input_a, input_b, input_c, target_y = sample[0],sample[1],sample[2], int(sample[3])
        a_data_word.append(list(map(lambda x:pre.getVector(x), input_a)))
        b_data_word.append(list(map(lambda x:pre.getVector(x), input_b)))
        c_data_word.append(input_c)
        if target_y == 1:
            y.append([0,1])
        else:
            y.append([1,0])


    a_data_word = kr.preprocessing.sequence.pad_sequences(np.array(a_data_word), model.config.X_maxlen)
    b_data_word = kr.preprocessing.sequence.pad_sequences(np.array(b_data_word), model.config.Y_maxlen)
    c_data_word = kr.preprocessing.sequence.pad_sequences(np.array(c_data_word), model.config.Y_maxlen)
    return a_data_word,b_data_word, c_data_word, np.array(y)

#采用one-hot形式来表示前后件信息,其中最后两个一定是前后件和数据label
def processInitData2(data,model):
    a_data_word = []
    b_data_word = []
    c_data_word = []
    y = []

    for sample in data:
        assert len(sample) == len(data), ValueError("the number of elemengs in this sample is {0}".format(len(sample)))
        input_a, input_b, input_c, target_y = sample[0],sample[1],sample[2], int(sample[3])
        a_data_word.append(list(map(lambda x:pre.getVector(x), input_a)))
        b_data_word.append(list(map(lambda x:pre.getVector(x), input_b)))

        c_line = []
        for c in input_c:
            assert c in [0,1],ValueError("label of qhj is wrong!" + str(c))
            if c == 0:
                c_line.append(np.array([1,0]))
            elif c == 1:
                c_line.append(np.array([0,1]))

        c_data_word.append(np.array(c_line))


        if target_y == 1:
            y.append([0,1])
        else:
            y.append([1,0])


    a_data_word = kr.preprocessing.sequence.pad_sequences(a_data_word, model.config.X_maxlen)
    b_data_word = kr.preprocessing.sequence.pad_sequences(b_data_word, model.config.Y_maxlen)
    c_data_word = kr.preprocessing.sequence.pad_sequences(c_data_word, model.config.Y_maxlen)
    return a_data_word,b_data_word, c_data_word, np.array(y)

def processInitData_Generic_OneHotQHJ(data,model):
    output_data = [[] for _ in range(len(data[0]))]

    for sample in data:
        #assert len(sample) == len(data), ValueError("the number of elemengs in this sample is {0}".format(len(sample)))
        for i, input in enumerate(sample[:-2]):
            output_data[i].append(list(map(lambda x:pre.getVector(x), input)))

        c_line = []
        for c in sample[-2]:
            assert c in [0,1],ValueError("label of qhj is wrong!" + str(c))
            if c == 0:
                c_line.append(np.array([1,0]))
            elif c == 1:
                c_line.append(np.array([0,1]))

        output_data[-2].append(np.array(c_line))


        if sample[-1] == 1:
            output_data[-1].append([0,1])
        else:
            output_data[-1].append([1,0])

    output_data[0] = kr.preprocessing.sequence.pad_sequences(output_data[0], model.config.X_maxlen)
    output_data[1] = kr.preprocessing.sequence.pad_sequences(output_data[1], model.config.Y_maxlen)
    output_data[2] = np.array(output_data[2])
    output_data[3] = kr.preprocessing.sequence.pad_sequences(output_data[3], model.config.Y_maxlen)
    output_data[4] = np.array(output_data[-1])
    return output_data
