#该文件的目标是获取三个数据，1、词向量 2、字向量 ==>3、输入数据的向量化表示，这些内容都存放在一个json文件中
import sys
import models.parameter as param
import processLaw as psLaw
import json
import jieba



#step prepare input
def setUp_inputs(trainPath = None, valPath = None, testPath = None):
    #read word info
    f_word = open(param.BaseConfig.w2vModel, 'r', encoding='utf-8')
    wordEmbedding = json.load(f_word)
    if '<UNK>' not in wordEmbedding.keys():
        wordEmbedding['<UNK>'] = '\t'.join(['0' for _ in range(param.BaseConfig.word_dimension)])
    wordVocab = wordEmbedding.keys()

    assert '<UNK>' in wordEmbedding.keys(), ValueError('space and unk not in word dict')
    assert len(wordVocab) == param.BaseConfig.word_vocab_size, ValueError('the number of word vocab is wrong, {0}'.format(len(wordVocab)))

    train = ""
    test = ""
    val = ""
    if trainPath:
       train = _setUp_inputs_(trainPath,wordEmbedding, wordVocab)
    if testPath:
       test = _setUp_inputs_(testPath, wordEmbedding, wordVocab)
    if valPath:
       val = _setUp_inputs_(valPath, wordEmbedding, wordVocab)
    env = {'train': train, 'test': test, 'val': val}
    return env


def _setUp_inputs_(sourcePath, wordEmbedding, wordVocab):
    with open(sourcePath,'r',encoding='utf-8') as fr:
        lines = fr.readlines()
    result = []
    for line in lines[:300]:
        line = line.strip()
        if line != '':
            items = line.split('|')
            assert len(items) == 4, ValueError("The number of items in this line is less than 4")
            fact_input = processText(items[1],wordEmbedding, wordVocab)
            law_input = processText(items[2],wordEmbedding, wordVocab)
            assert items[3] in ['0', '1'], ValueError("Label is not in [0,1]!")
            label = items[3]
            result.append([fact_input, law_input, label])
    return result
import re
from util.multitest import MultiProcess
from multiprocessing import Queue,Pool
import time
q1 = Queue()
q2 = Queue()
q3 = Queue()

def setUp_inputs_QHJ(trainPath = None, valPath = None, testPath = None, rfModel=None):
    #read word info
    f_word = open(param.BaseConfig.w2vModel, 'r', encoding='utf-8')
    wordEmbedding = json.load(f_word)
    if '<UNK>' not in wordEmbedding.keys():
        wordEmbedding['<UNK>'] = '\t'.join(['0' for _ in range(param.BaseConfig.word_dimension)])
    wordVocab = wordEmbedding.keys()

    assert '<UNK>' in wordEmbedding.keys(), ValueError('space and unk not in word dict')
    assert len(wordVocab) == param.BaseConfig.word_vocab_size, ValueError('the number of word vocab is wrong, {0}'.format(len(wordVocab)))

    train = []
    test = []
    val = []
    if trainPath:
        # args = []
        # tars = []
        # for i in range(15):
        #     args.append((trainPath,wordEmbedding, wordVocab, rfModel,i * 1000,i * 1000 + 1000,1))
        #     tars.append(_setUp_inputs_QHJ)
        # mp = MultiProcess(tar=tars,arg=args)
        # mp.multi_processing()
        # while not q1.empty():
        #     train += list(q1.get())

        train = _setUp_inputs_QHJ_lawone(trainPath, wordEmbedding, wordVocab, rfModel, 0, 15000,1)

    if valPath:
        # args = []
        # tars = []
        # for i in range(10):
        #     args.append((valPath, wordEmbedding, wordVocab, rfModel, i * 100, i * 100 + 100,3))
        #     tars.append(_setUp_inputs_QHJ)
        # mp = MultiProcess(tar=tars, arg=args)
        # mp.multi_processing()
        # while not q3.empty():
        #     val += list(q3.get())

        val = _setUp_inputs_QHJ_KS(valPath, wordEmbedding, wordVocab, rfModel, 0, 1000, 3)

    if testPath:
        # args = []
        # tars = []
        # for i in range(10):
        #     args.append((testPath, wordEmbedding, wordVocab, rfModel, i * 100, i * 100 + 100,2))
        #     tars.append(_setUp_inputs_QHJ)
        # mp = MultiProcess(tar=tars, arg=args)
        # mp.multi_processing()
        # while not q2.empty():
        #     test += list(q2.get())

        test = _setUp_inputs_QHJ_KS(testPath, wordEmbedding, wordVocab, rfModel, 0, 1000, 2)

    env = {'train': train, 'test': test, 'val': val}
    return env

#只使用前件或者后件的正文
def setUp_inputs_QHJ_lawone(trainPath = None, valPath = None, testPath = None, rfModel=None,flag=0):
    #read word info
    f_word = open(param.BaseConfig.w2vModel, 'r', encoding='utf-8')
    wordEmbedding = json.load(f_word)
    if '<UNK>' not in wordEmbedding.keys():
        wordEmbedding['<UNK>'] = '\t'.join(['0' for _ in range(param.BaseConfig.word_dimension)])
    wordVocab = wordEmbedding.keys()

    assert '<UNK>' in wordEmbedding.keys(), ValueError('space and unk not in word dict')
    assert len(wordVocab) == param.BaseConfig.word_vocab_size, ValueError('the number of word vocab is wrong, {0}'.format(len(wordVocab)))

    train = []
    test = []
    val = []
    if trainPath:
        train = _setUp_inputs_QHJ_lawone(trainPath, wordEmbedding, wordVocab, rfModel, 0, 1500,flag)

    if valPath:
        val = _setUp_inputs_QHJ_lawone(valPath, wordEmbedding, wordVocab, rfModel, 0, 1000, flag)

    if testPath:
        test = _setUp_inputs_QHJ_lawone(testPath, wordEmbedding, wordVocab, rfModel, 0, 1000, flag)

    env = {'train': train, 'test': test, 'val': val}
    return env

def _setUp_inputs_QHJ_lawone(sourcePath, wordEmbedding, wordVocab,rfModel,start,end,flag):
    stp = list(map(lambda x: x.strip(), open(param.BaseConfig.stpPath, 'r', encoding='utf-8').read().split('\n')))

    with open(sourcePath,'r',encoding='utf-8') as fr:
        lines = fr.readlines()
    result = []
    count = 0
    if start >= len(lines):
        return
    end = min(len(lines), end)

    for line in lines[start:end]:
        line = line.strip()
        if line != '':
            items = line.split('|')
            assert len(items) == 4, ValueError("The number of items in this line is less than 4, content:" + line)
            fact_input = processTextWithStpDict(items[1],wordEmbedding, wordVocab,stp)
            if len(fact_input) == 0: continue
            law_units = items[2].split(':')
            law_name = law_units[0]
            law_content = items[2][len(law_name) + 1:]
            law_content, law_input_vector = psLaw.processLawForRf(law_content)
            #接下来预测每句话的label,并将其映射到每个词上
            law_labels = rfModel.predict(law_input_vector)
            content_split = re.split(r"[，；。：]",law_content)
            content_split = list(filter(lambda x: x != "", list(map(lambda x: x.strip(), content_split))))
            law_input = []
            assert len(content_split) == len(law_labels), ValueError("content_split:{0}, law_label:{1}, line:{3}".format(len(content_split), len(law_labels), line))
            for law_label,content in zip(law_labels,content_split):
                if law_label == flag:
                    content_vector = processTextWithoutDict(content, wordEmbedding, wordVocab)
                    law_input.extend(content_vector)
            assert items[3] in ['0', '1'], ValueError("Label is not in [0,1]!")
            label = items[3]
            result.append([fact_input, law_input, label])
            count += 1
            print("precessing {0}/{1} samples".format(count,len(lines)))
    return result

def _setUp_inputs_QHJ(sourcePath, wordEmbedding, wordVocab,rfModel,start,end,flag):
    stp = list(map(lambda x: x.strip(), open(param.BaseConfig.stpPath, 'r', encoding='utf-8').read().split('\n')))
    with open(sourcePath,'r',encoding='utf-8') as fr:
        lines = fr.readlines()
    result = []
    count = 0
    if start >= len(lines):
        return
    end = min(len(lines), end)

    for line in lines[start:end]:
        line = line.strip()
        if line != '':
            items = line.split('|')
            assert len(items) == 4, ValueError("The number of items in this line is less than 4, content:" + line)
            fact_input = processTextWithStpDict(items[1],wordEmbedding, wordVocab,stp)
            if len(fact_input) == 0:
                if flag != 2:
                   print("Ignore fact!")
                   continue
                else:
                    fact_input = [[0 for _ in range(len(param.BaseConfig.word_dimension))]]
                    print("None fact")

            law_units = items[2].split(':')
            law_name = law_units[0]
            law_content = items[2][len(law_name) + 1:]
            law_content, law_input_vector = psLaw.processLawForRf(law_content)
            #接下来预测每句话的label,并将其映射到每个词上
            law_labels = rfModel.predict(law_input_vector)
            content_split = re.split(r"[，；。：]",law_content)
            content_split = list(filter(lambda x: x != "", list(map(lambda x: x.strip(), content_split))))
            law_input = []
            law_label_input = []
            assert len(content_split) == len(law_labels), ValueError("content_split:{0}, law_label:{1}, line:{3}".format(len(content_split), len(law_label), line))
            for law_label,content in zip(law_labels,content_split):
                content_vector = processTextWithoutDict(content, wordEmbedding,wordVocab)
                law_input.extend(content_vector)
                law_label_input+= [law_label for _ in range(len(content_vector))]
            assert len(law_input) == len(law_label_input), ValueError("law:{0},label:{1}".format(len(law_input),len(law_label_input)))
            assert items[3] in ['0', '1'], ValueError("Label is not in [0,1]!")
            label = items[3]
            result.append([fact_input, law_input, law_label_input, label])
            count += 1
            print("precessing {0}/{1} samples".format(count,len(lines)))
    return result

#不添加前后件信息
def _setUp_inputs_QHJ_2(sourcePath, wordEmbedding, wordVocab,rfModel,start,end,flag):
    stp = list(map(lambda x: x.strip(), open(param.BaseConfig.stpPath, 'r', encoding='utf-8').read().split('\n')))

    with open(sourcePath,'r',encoding='utf-8') as fr:
        lines = fr.readlines()
    result = []
    count = 0
    if start >= len(lines):
        return
    end = min(len(lines), end)

    for line in lines[start:end]:
        line = line.strip()
        if line != '':
            items = line.split('|')
            assert len(items) == 4, ValueError("The number of items in this line is less than 4, content:" + line)
            fact_input = processTextWithStpDict(items[1],wordEmbedding, wordVocab,stp)
            if len(fact_input) == 0: continue
            law_units = items[2].split(':')
            law_name = law_units[0]
            law_content = items[2][len(law_name) + 1:]
            law_content, law_input_vector = psLaw.processLawForRf(law_content)
            #接下来预测每句话的label,并将其映射到每个词上
            content_split = re.split(r"[，；。：]",law_content)
            content_split = list(filter(lambda x: x != "", list(map(lambda x: x.strip(), content_split))))
            law_input = []
            for content in content_split:
                content_vector = processTextWithoutDict(content, wordEmbedding,wordVocab)
                law_input.extend(content_vector)
            assert items[3] in ['0', '1'], ValueError("Label is not in [0,1]!")
            label = items[3]
            result.append([fact_input, law_input, label])
            count += 1
            print("precessing {0}/{1} samples".format(count,len(lines)))
    return result

def setUp_inputs_QHJ_lawtwo(trainPath = None, valPath = None, testPath = None, rfModel=None,flag=0):
    #read word info
    f_word = open(param.BaseConfig.w2vModel, 'r', encoding='utf-8')
    wordEmbedding = json.load(f_word)
    if '<UNK>' not in wordEmbedding.keys():
        wordEmbedding['<UNK>'] = '\t'.join(['0' for _ in range(param.BaseConfig.word_dimension)])
    wordVocab = wordEmbedding.keys()

    assert '<UNK>' in wordEmbedding.keys(), ValueError('space and unk not in word dict')
    assert len(wordVocab) == param.BaseConfig.word_vocab_size, ValueError('the number of word vocab is wrong, {0}'.format(len(wordVocab)))

    train = []
    test = []
    val = []
    if trainPath:
        train = _setUp_inputs_QHJ_split(trainPath, wordEmbedding, wordVocab, rfModel, 0, 15000,flag)

    if valPath:
        val = _setUp_inputs_QHJ_split(valPath, wordEmbedding, wordVocab, rfModel, 0, 1000, flag)

    if testPath:
        test = _setUp_inputs_QHJ_split(testPath, wordEmbedding, wordVocab, rfModel, 0, 1000, flag)

    env = {'train': train, 'test': test, 'val': val}
    return env
#不添加法条的前后件信息，将法条的前件作为一个输入，法条的后件作为一个输入
def _setUp_inputs_QHJ_split(sourcePath, wordEmbedding, wordVocab,rfModel,start,end,flag):
    stp = list(map(lambda x:x.strip(),open(param.BaseConfig.stpPath,'r',encoding='utf-8').read().split('\n')))

    with open(sourcePath,'r',encoding='utf-8') as fr:
        lines = fr.readlines()
    result = []
    count = 0
    if start >= len(lines):
        return
    end = min(len(lines), end)

    for line in lines[start:end]:
        line = line.strip()
        if line != '':
            items = line.split('|')
            assert len(items) == 4, ValueError("The number of items in this line is less than 4, content:" + line)
            fact_input = processTextWithStpDict(items[1],wordEmbedding, wordVocab,stp)
            if len(fact_input) == 0: continue
            law_units = items[2].split(':')
            law_name = law_units[0]
            law_content = items[2][len(law_name) + 1:]
            law_content, law_input_vector = psLaw.processLawForRf(law_content)
            #接下来将法条的前后件拆分放在不同列表里
            law_labels = rfModel.predict(law_input_vector)
            content_split = re.split(r"[，；。：]",law_content)
            content_split = list(filter(lambda x: x != "", list(map(lambda x: x.strip(), content_split))))
            law_input_qj = []
            law_input_hj = []
            assert len(content_split) == len(law_labels), ValueError("content_split:{0}, law_label:{1}, line:{3}".format(len(content_split), len(law_labels), line))
            for law_label,content in zip(law_labels,content_split):
                content_vector = processTextWithoutDict(content, wordEmbedding, wordVocab)
                if law_label == 0:
                    law_input_qj.extend(content_vector)
                else:
                    law_input_hj.extend(content_vector)

            assert items[3] in ['0', '1'], ValueError("Label is not in [0,1]!")
            label = items[3]
            result.append([fact_input, law_input_qj, law_input_hj, label])
            count += 1
            print("precessing {0}/{1} samples".format(count,len(lines)))
    return result

def processTextWithoutDict(line,wordEmbedding, wordVocab):
    initContent = line.strip()
    if initContent != "":
        content = jieba.cut(initContent)
        lines = list(map(lambda x: str(x).strip(), content))
        contentcut = list(filter(lambda x: x != "", lines))
        wordEmbs = []
        for word in contentcut:
            wordEmb = processWord(word,wordEmbedding,wordVocab)
            wordEmbs.append(wordEmb)
        return wordEmbs
    return []

#只保留特定词性的词
import jieba.posseg as pos
def processTextWithStpDict(line,wordEmbedding, wordVocab,stp):
    initContent = line.strip()
    if initContent != "":
        content = pos.cut(initContent)
        contentcut = []
        for w in content:
            word = w.word.strip()
            if word == "" or word in stp or w.flag in ['nr','ns','p','u']: continue
            contentcut.append(word)

        if len(contentcut) == 0: return []
        wordEmbs = []
        for word in contentcut:
            wordEmb = processWord(word,wordEmbedding,wordVocab)
            wordEmbs.append(wordEmb)
        return wordEmbs
    return []


def processText(line,wordEmbedding, wordVocab):
    initContent = line.strip()
    if initContent != "":
        content = jieba.cut(initContent)
        lines = list(map(lambda x: str(x).strip(), content))
        contentcut = list(filter(lambda x: x != "", lines))
        wordEmbs = []
        for word in contentcut:
            wordEmb = processWord(word,wordEmbedding,wordVocab)
            wordEmbs.append(wordEmb)
        return {'word_input': wordEmbs}
    return []

def processLawText(line,wordEmbedding, wordVocab):
    initContent = line.strip()
    if initContent != "":
        content = jieba.cut(initContent)
        lines = list(map(lambda x: str(x).strip(), content))
        contentcut = list(filter(lambda x: x != "", lines))
        wordEmbs = []
        for word in contentcut:
            wordEmb = processWord(word, wordEmbedding, wordVocab)
            #判断是否是NoneType
            if wordEmb is None: continue
            wordEmb = getVector(wordEmb)
            if wordEmb is None: continue
            wordEmbs.append(wordEmb)
        return wordEmbs
    return []


def getVector(str_vector):
    vectors = str_vector.split('\t')
    vectors = list(map(float, map(lambda x:x.strip(),filter(lambda x: x.strip() != '', vectors))))
    return vectors

def processWord(word, word_embedding, vocabs):
    if word not in vocabs:
        return word_embedding['<UNK>']
    else:
        return word_embedding[word]

#下面是处理具有先验知识===============================
import numpy as np
def _setUp_inputs_QHJ_KS(sourcePath, wordEmbedding, wordVocab,rfModel,start,end,flag):
    stp = open(param.BaseConfig.stpPath,'r',encoding='utf-8').read().split('\n')
    fr_lawks = open(param.BaseConfig.lawKsPath,'r',encoding='utf-8')
    lawks_dict = json.load(fr_lawks)

    with open(sourcePath,'r',encoding='utf-8') as fr:
        lines = fr.readlines()
    result = []
    count = 0
    if start >= len(lines):
        return
    end = min(len(lines), end)

    for line in lines[start:end]:
        line = line.strip()
        if line != '':
            items = line.split('|')
            assert len(items) == 4, ValueError("The number of items in this line is less than 4, content:" + line)
            fact_input = processTextWithStpDict(items[1],wordEmbedding, wordVocab,stp)
            if len(fact_input) == 0: continue
            law_units = items[2].split(':')
            law_name = law_units[0]
            law_content = items[2][len(law_name) + 1:]
            law_content, law_input_vector = psLaw.processLawForRf(law_content)
            #接下来预测每句话的label,并将其映射到每个词上
            law_labels = rfModel.predict(law_input_vector)
            content_split = re.split(r"[，；。：]",law_content)
            content_split = list(filter(lambda x: x != "", list(map(lambda x: x.strip(), content_split))))
            law_input = []
            law_label_input = []
            assert len(content_split) == len(law_labels), ValueError("content_split:{0}, law_label:{1}, line:{3}".format(len(content_split), len(law_labels), line))
            for law_label,content in zip(law_labels,content_split):
                content_vector = processTextWithoutDict(content, wordEmbedding,wordVocab)
                law_input.extend(content_vector)
                law_label_input+= [law_label for _ in range(len(content_vector))]
            assert len(law_input) == len(law_label_input), ValueError("law:{0},label:{1}".format(len(law_input),len(law_label_input)))
            assert items[3] in ['0', '1'], ValueError("Label is not in [0,1]!")
            label = items[3]

            #添加先验知识
            law_ks = lawks_dict[law_name]
            law_ks_vector = []

            for ks in law_ks:
                if ks == '': ks_vector = [0 for _ in range(param.BaseConfig.word_dimension)]
                else:
                    vector_str = processTextWithoutDict(ks,wordEmbedding,wordVocab)
                    vectors = []
                    for v in vector_str:
                       vectors.append(getVector(v))
                    ks_vector = np.mean(np.array(vectors), axis=0)
                law_ks_vector.append('\t'.join(list(map(str,ks_vector))))
            result.append([fact_input, law_input, law_ks_vector, law_label_input, label])
            count += 1
            print("precessing {0}/{1} samples".format(count,len(lines)))
    return result







