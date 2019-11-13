#该文件的目标是获取三个数据，1、词向量 2、字向量 ==>3、输入数据的向量化表示，这些内容都存放在一个json文件中
import sys
sys.path.append('../models')
import models.parameter as param
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
    for line in lines:
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


def getVector(str_vector):
    vectors = str_vector.split('\t')
    vectors = list(map(float, map(lambda x:x.strip(),filter(lambda x: x.strip() != '', vectors))))
    return vectors

def processWord(word, word_embedding, vocabs):
    if word not in vocabs:
        return word_embedding['<UNK>']
    else:
        return word_embedding[word]




















