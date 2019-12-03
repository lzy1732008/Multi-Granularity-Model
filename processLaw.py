import re
import json
import os
from zhon.hanzi import punctuation
import jieba

def getAllLaw(datapath):
    laws = {}
    fr = open(datapath,'r',encoding='utf-8')
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        array = line.split('|')
        ft = array[1].strip()
        if ft == "":
            continue
        pattern = '\(\d{4}\)'
        regx = re.compile(pattern)
        res = regx.split(ft)
        assert len(res) == 2, ValueError("split is wrong, it is " + ft)
        law = res[0]
        num = res[1]
        if law not in laws.keys():
            laws[law] = [num]
        else:
            laws[law].append(num)

    print("The article of each law")
    laws = sorted(laws.items(), key=lambda x:len(x[1]),reverse=True)
    for k,v in laws:
        print(k,len(v))

def getFtQH(datapath):
    #加载已经分好的法条前后件
    prefr = open('法条前后件.json', 'r', encoding='utf-8')
    preRes = json.load(prefr)

    fr = open(datapath, 'r', encoding='utf-8')
    storePath = 'resource/ftSplitStore.json'
    if os.path.exists(storePath):
        print("文件已存在！请输出新文件序号！")
        storePath = input()
        storePath = 'resource/ftSplitStore'+storePath+'.json'
    fw = open(storePath,'w',encoding='utf-8')
    lines = fr.readlines()
    ft_dict = {}
    pattern = '，|。|；|：'
    regx = re.compile(pattern)
    unSaved = []

    for index,line in enumerate(lines[:15]):
        print("处理第{0}条法条".format(index))
        line = line.strip()
        if line == "":
            continue
        array = line.split('|')
        assert len(array) == 3,ValueError("Error label"+line)
        article = array[1]
        content = array[2]
        ft_dict[article] = []
        print("current article.......")
        print(article+"："+content)
        contentSplit = regx.split(content)

        pattern_article = '\(\d{4}\)'
        regx_article = re.compile(pattern_article)
        res_article = regx_article.split(article)
        assert len(res_article) == 2, json.dump(ft_dict, fw) and ValueError("Wrong Input!")
        law = res_article[0]
        num = res_article[1]
        j = 0
        enterCount = 0
        while j < len(contentSplit):
            con = contentSplit[j]
            con = con.strip()
            if con == "":
                j += 1
                continue

            print("label items.....")
            print("id:{0}, 正文:{1}".format(j,con))

            #首先判断是否已经有标注了
            ifLabeled = False
            if law+":"+num in preRes.keys():
                currentStr = []
                cIndex = j
                while cIndex < len(contentSplit) and cIndex < j + 6:
                    currentStr.append(contentSplit[cIndex])
                    newStr = '，'.join(currentStr)
                    label = -1
                    if newStr in preRes[law+":"+num]["Qs"] or newStr[:-1] in preRes[law+":"+num]["Qs"]:
                        label = 0
                    elif newStr in preRes[law+":"+num]["Hs"] or newStr[:-1] in preRes[law+":"+num]["Hs"]:
                        label = 1
                    if label != -1:
                        for e in currentStr:
                            print("{0}:为已被标注，label:{1}".format(e,str(label)))
                            ft_dict[article].append((e,label))
                            j += 1
                        ifLabeled = True
                        continue
                    else:
                        cIndex += 1

            if ifLabeled: continue


                # if con in preRes[law+":"+num]["Qs"] or con[:-1] in preRes[law+":"+num]["Qs"]:
                #     print("为已被标注的前件......")
                #     label = 0
                #     ft_dict[article].append((con, label))
                #     j += 1
                #     continue
                # elif con in preRes[law+":"+num]["Hs"]:
                #     print("为已被标注的后件......")
                #     label = 1
                #     ft_dict[article].append((con, label))
                #     j += 1
                #     continue
                # #判断是否是一个前件的子串
                # elif j + 1 < len(contentSplit) and (con + "，"+contentSplit[j + 1][:-1] in preRes[law+":"+num]["Qs"] or
                #                                     con + "，"+contentSplit[j + 1] in preRes[law+":"+num]["Qs"]):
                #     print("为已被标注的前件的子件")
                #     label = 0
                #     ft_dict[article].append((con, label))
                #     j += 1
                #     print("标注下一个前件子件")
                #     ft_dict[article].append((contentSplit[j + 1], label))
                #     j += 1
                #     continue
                # elif j + 1 < len(contentSplit) and con +'，'+contentSplit[j + 1] in preRes[law+":"+num]["Hs"]:
                #     print("为已被标注的后件的子件")
                #     label = 1
                #     ft_dict[article].append((con, label))
                #     j += 1
                #     print("标注下一个后件子件")
                #     ft_dict[article].append((contentSplit[j + 1], label))
                #     j += 1
                #     continue
                # elif j + 2 < len(contentSplit) and (con + "，"+contentSplit[j + 1] + "，" +contentSplit[j + 2][:-1] in preRes[law+":"+num]["Qs"]
                #                                     or con + "，"+contentSplit[j + 1] + "，"+ contentSplit[j + 2] in preRes[law+":"+num]["Qs"]):
                #     print("为已被标注的前件的子件")
                #     label = 0
                #     ft_dict[article].append((con, label))
                #     j += 1
                #     print("标注下一个前件子件")
                #     ft_dict[article].append((contentSplit[j + 1], label))
                #     j += 1
                #     print("标注下一个前件子件")
                #     ft_dict[article].append((contentSplit[j + 1], label))
                #     j += 1
                #     continue
                # elif j + 2 < len(contentSplit) and con +'，'+contentSplit[j + 1]+'，' in preRes[law+":"+num]["Hs"]:
                #     print("为已被标注的后件的子件")
                #     label = 1
                #     ft_dict[article].append((con, label))
                #     j += 1
                #     print("标注下一个后件子件")
                #     ft_dict[article].append((contentSplit[j + 1], label))
                #     j += 1
                #     print("标注下一个后件子件")
                #     ft_dict[article].append((contentSplit[j + 1], label))
                #     j += 1
                #     continue

            print('please label...')
            enterCount += 1
            label = input()  # 前件代表0，后件代表1，无关词语代表2,3代表停止
            if label == "4":
                unSaved.append(article+":"+con)
                print("当前法条内容尚不明确分类！")
                continue
            elif label not in ["0","1","2"]:
                json.dump(ft_dict, fw)
                print("处理到第{0}个法条，第{1}个子项".format(index, j))
                return
            ft_dict[article].append((con, label))
            j += 1

    json.dump(ft_dict, fw)
    print("恭喜！全部标注完毕！！")
    print('\n'.join(unSaved))
    print("总共标注了{0}个数据".format(enterCount))


def setFTQHStore(datapath):
    laws = {}
    pattern = '^第(（(一|二|三|四|五|六|七|八|九|十)）|(一|二|三|四|五|六|七|八|九|十))(款|项)$'
    regx = re.compile(pattern)

    fr = open(datapath, 'r', encoding='utf-8')
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        array = line.split()
        ftname = array[0][1:-1]
        article = array[4]
        if ftname+":"+article not in laws.keys():
            laws[ftname+":"+article] = {"Qs":[],"Hs":[]}
        laws[ftname + ":" + article]["Hs"].append(array[-1])
        if regx.match(array[-2]) is None:
            laws[ftname + ":" + article]["Qs"].append(array[-2])
    fw = open('法条前后件.json','w',encoding='utf-8')
    json.dump(laws,fw)

# datapath = 'resource/刑法前后件结果-人工调整.txt'
# setFTQHStore(datapath)
#
# fr = open('法条前后件.json','r',encoding='utf-8')
# res = json.load(fr)
# print(res["中华人民共和国刑事诉讼法:第二百八十一条"]["Hs"])

#
# datapath = 'resesource/ft_nr_QHSplit.txt'
# getFtQH(datapath)

def checkAllLabeled(targetpath):
    file_nums = ['','15','104','141','149','164','198','262','304','311','339','344','372','433']
    fw = open(targetpath,'w',encoding='utf-8')
    allres = {}
    for num in file_nums:
        file_name = 'resource/ftSplitStore'+num+'.json'
        fr = open(file_name,'r',encoding='utf-8')
        res = json.load(fr)
        for ft, content in res.items():
            allres[ft] = content

    output = ""
    for ft, content in allres.items():
        content = list(map(lambda x: x[0]+'\t'+str(x[1]),content))
        ft_output = "法条:{0}\n{1}\n".format(ft,'\n'.join(content))
        output += ft_output
    fw.write(output)

# targetPath = "resource/ftSplitCheck.txt"
# checkAllLabeled(targetPath)

#填充标点符号
def fillFH():
    # sourceLaw = 'resource/ft_nr_QHSplit.txt'
    # labeledLaw = 'resource/ftSplitCheck.txt'
    # targetLaw = 'resource/ft_labeled.json'
    sourceLaw = 'resource/gyshz_ftchina_zw.txt'
    labeledLaw = 'resource/gyshz_split_labeled.txt'
    targetLaw = 'resource/gyshz_ft_labeled.json'
    fr1 = open(sourceLaw,'r',encoding='utf-8')
    fr2 = open(labeledLaw,'r',encoding='utf-8')
    fw = open(targetLaw,'w',encoding='utf-8')
    source1_dict = {}
    source2_dict = {}
    #获取第一个源文件的内容
    lines = fr1.readlines()
    # for line in lines:
    #     line = line.strip()
    #     if line == "":
    #         continue
    #     array = line.split('|')
    #     ft = array[1].strip()
    #     content = array[2].strip()
    #     if ft == "" or content == "":
    #         continue
    #
    #     source1_dict[ft] = content
    index = 0
    while index + 1 < len(lines):
        ft = lines[index][3:].strip()
        content = lines[index + 1].strip()
        source1_dict[ft] = content
        index += 2



    lines = fr2.read().split('法条:')
    for line in lines:
        array = line.split('\n')
        labeledDatas = []
        for c in array[1:]:
            c = c.strip()
            if c == "": continue
            # labeled = c.split('\t')
            labeled = c.split()
            assert len(labeled) == 2, ValueError("Wrong split labeled line:" + c)
            assert labeled[-1] in ['0','1','2'],ValueError("Wrong labeled line:" + c)
            labeledDatas.append(labeled)
        source2_dict[array[0]] = labeledDatas


    #将两者合并
    output_dict = {}
    for ft, content in source1_dict.items():
        labeledContent = source2_dict[ft]
        contents = []
        for subitem in labeledContent:
            assert content != "", ValueError("content has already been null, the subitem is "+ subitem)
            startIndex = str(content).find(subitem[0])
            endIndex = startIndex + len(subitem[0])
            assert startIndex != -1, ValueError("Find no substring matches for subitem {0} in law {1}!".format(subitem,ft))
            assert endIndex < len(content), ValueError("Wrong split" + subitem[0])
            assert str(content[endIndex]) in punctuation, ValueError("End char isn't punctuation!"+subitem[0])
            contents.append([content[startIndex:endIndex+1],subitem[1]])
            content = content[endIndex+1:]
        output_dict[ft] = contents
    json.dump(output_dict,fw)

# fillFH()

import random
import numpy as np

def buildDataSet(window = 1):
    jieba.load_userdict('resource/dict.txt')
    fw = open('resource/gyshz_lawDataSet.json','w',encoding='utf-8')
    dataset = {}
    fr = open('resource/gyshz_ft_labeled.json','r',encoding='utf-8')
    allft = json.load(fr).items()
    alldata = []
    for ft,labeledcontents in allft:
        for i, c in enumerate(labeledcontents):
            #这一段用于删除类别为2的文字预测 *****************
            if int(c[1]) == 2:
                continue
            # 这一段用于删除类别为2的文字预测 *****************

            currentData = ""
            if i == 0:
                currentData += 'S'
            else:
                for j in range(i-window,i):currentData += labeledcontents[j][0]

            currentData += c[0]

            if i == len(labeledcontents) - 1:
                currentData += 'E'
            else:
                for j in range(i+1,i+window+1): currentData += labeledcontents[j][0]
            #进行分词
            alldata.append((currentData,c[1]))
            # alldata.append((' '.join(jieba.lcut(currentData)), c[1]))

    indices = np.random.permutation(np.arange(len(alldata)))
    alldata_ = np.array(alldata)[indices].tolist()
    numOfTrain = int(len(alldata) * 0.8)
    numOfVal = int(len(alldata) * 0.1)
    trainSet = alldata_[:numOfTrain]
    valSet = alldata_[numOfTrain:numOfTrain + numOfVal]
    testSet = alldata_[numOfTrain + numOfVal:]
    dataset["dataSet"] = trainSet
    json.dump(dataset,fw)

# buildDataSet()

import models.parameter as param
def extendWordVocabs():
    f_word = open(param.BaseConfig.w2vModel, 'r', encoding='utf- 8')
    wordEmbedding = json.load(f_word)
    if '<UNK>' not in wordEmbedding.keys():
        wordEmbedding['<UNK>'] = '\t'.join(['0' for _ in range(param.BaseConfig.word_dimension)])
    newWords = ['S', 'E'] + list(punctuation)
    newWordsEmbedding = np.reshape(np.random.randn(len(newWords) * param.BaseConfig.word_dimension),
                                   newshape=[len(newWords), param.BaseConfig.word_dimension])
    for i, nw in enumerate(newWords):
        wordEmbedding[nw] = '\t'.join(map(str,newWordsEmbedding[i].tolist()))

    w_word = open(param.BaseConfig.w2vModel_ex, 'w', encoding='utf-8')
    json.dump(wordEmbedding,w_word)

# extendWordVocabs()



#下面是为了传统机器学习分类器做法条预处理
#首先建立法条正文文本的字典
def buildLawDict():
    dictionary = {}
    fr = open('resource/corpus_law.txt','r',encoding='utf-8')
    lines = fr.readlines()
    for line in lines:
        words = list(map(lambda x: x.strip(),list(filter(lambda x: x.strip()!="",line.split()))))
        for w in words:
            if w in dictionary.keys():
                dictionary[w] += 1
            else:
                dictionary[w] = 1
    dictionary = sorted(dictionary.items(),key=lambda x:x[1],reverse=True)
    remainedWords = []
    for w, n in dictionary:
        if n > 40:
            remainedWords.append(w)
    print('\n'.join(remainedWords))
    print('总共词语为：'+str(len(remainedWords)))


# buildLawDict()
from util import rules
def processLawText2id(line,dictionary):
    initContent = line.strip()
    vector = []
    if initContent != "":
        for word in dictionary:
            times = 0
            if str(initContent).count(word) > 0: times = 1
            vector.append(times)
        return vector
    return []

def ruleFeatures(pre,cText,next):
    pre_QRule = rules.QRules()
    pre_HRule = rules.HRules()
    cText_QRule = rules.QRules()
    cText_HRule = rules.HRules()
    next_QRule = rules.QRules()
    next_HRule = rules.HRules()

    #判断当前和之后的句子是否以连词开头
    # cText_conRule = rules.Rules(cText)
    # next_conRule = rules.Rules(next)
    # con_words = ["并且","或者","但是","并","而且"]
    return pre_QRule.inter(pre) + pre_HRule.inter(pre) + cText_QRule.inter(cText) + cText_HRule.inter(cText) + next_QRule.inter(next) + next_HRule.inter(next)
           # + [int(cText_conRule.rule1(con_words)),int(next_conRule.rule1(con_words))]



def processMultiLawText2id(line,dictionary):
    pattern = '，|。|；|：'
    regx = re.compile(pattern)
    array = regx.split(line)
    pre, cText, next = '', '', ''
    puncVectors = []
    punction = {"，":1,'。':2,'：':3,"；":4}

    assert len(array) in [2,3,4], ValueError("Contain wrong number of sub items:{0} with {1} sub items".format(input,len(array)))
    if len(array) == 2: #独立的一个句子
        v1 = [0 for _ in range(len(dictionary) - 2)] + [1, 0]
        v2 = processLawText2id(array[0][1:],dictionary)
        v3 = [0 for _ in range(len(dictionary) - 1)] + [1]

        #前后都是空
        cText = array[0][1:]

        #标点符号
        puncVectors = [0,punction[line[-2]],0]


    elif len(array) == 3: #
        if line[0] == 'S':
            v1 = [0 for _ in range(len(dictionary) - 2)] + [1, 0]
            v2 = processLawText2id(array[0][1:], dictionary)
            v3 = processLawText2id(array[1], dictionary)

            #前面是空
            cText = array[0][1:]
            next = array[1]

            #标点符号
            puncVectors = [0, punction[line[len(array[0])]], punction[line[-1]]]
        elif line[-1] == 'E':
            v1 = processLawText2id(array[0],dictionary)
            v2 = processLawText2id(array[1], dictionary)
            v3 = [0 for _ in range(len(dictionary) - 1)] + [1]

            #后面是空
            pre = array[0]
            cText = array[1]

            #标点符号
            puncVectors = [punction[line[len(array[0])]], punction[line[-2]], 0]

    else:
        v1 = processLawText2id(array[0], dictionary)
        v2 = processLawText2id(array[1], dictionary)
        v3 = processLawText2id(array[2], dictionary)

        #没有是空
        pre = array[0]
        cText = array[1]
        next = array[2]

        #标点符号
        # print("line len:{0}, array[0]:{1}, array[1]:{2}".format(len(line),len(array[0]),len(array[1])))
        puncVectors = [punction[line[len(array[0])]], punction[line[len(array[0])+len(array[1])+1]], \
                       punction[line[-1]]]


    assert len(v1) == len(v2) == len(v3), ValueError("Wrong vector size:" + line)
    rulefeatures = ruleFeatures(pre,cText,next)
    return v2 + rulefeatures + puncVectors



def buildDataSetForRF(train,val,test):
    fr = open('../resource/gyshz_lawDataSet.json', 'r', encoding='utf-8')
    env = json.load(fr)
    fr_dict = open('../resource/lawdict>30.txt','r',encoding='utf-8')
    dictionary = list(map(lambda x:x.strip(),fr_dict.readlines()))

    X = []
    Y = []
    if train:
        trainSet = env['trainSet']
        x,y = vectorText(trainSet,dictionary)
        X.extend(x)
        Y.extend(y)
    if val:
        valSet = env['valSet']
        x, y = vectorText(valSet, dictionary)
        X.extend(x)
        Y.extend(y)
    if test:
        testSet = env['dataSet']
        x, y = vectorText(testSet, dictionary)
        X.extend(x)
        Y.extend(y)
    return X,Y

def vectorText(data,dictionary):
    y = []
    inputs = []
    for index,sample in enumerate(data):
        input, target_y = sample[0], int(sample[1])

        if target_y == 2:
            continue
        vector = processMultiLawText2id(input,dictionary)
        inputs.append(vector)
        y.append(target_y)
    return [inputs,y]



def buildlabelfile(datapath):
    fr = open(datapath,'r',encoding='utf-8')
    lines = fr.readlines()
    output = ""
    pattern = '，|。|；|：'
    regx = re.compile(pattern)
    for line in lines:
        if line.startswith("法条:"):
            output += line
        else:
            array = regx.split(line)
            array = list(filter(lambda x: x != "",list(map(lambda x: x.strip(), array))))
            output += '\n'.join(array) + '\n'

    fw = open('resource/gyshz_split_labeled.txt','w',encoding='utf-8')
    fw.write(output)
    print('Done!')

# buildlabelfile('resource/gyshz_ftchina_zw.txt')


# 下列方法是为了使用随机森林模型进行预测进行的数据预处理
def processLawForRf(lawText):
    pattern = r'([，。；：])'
    fr_dict = open(param.BaseConfig.rf_dict_path, 'r', encoding='utf-8')
    dictionary = list(map(lambda x: x.strip(), fr_dict.readlines()))
    #首先是进行切割，然后构建<pre,text,next>这样的输入
    #首先根据原文把标点符号带上
    outputTexts = str(lawText).replace("其中，","").replace("但是，","")
    texts = re.split(pattern,outputTexts)
    texts = list(filter(lambda x: x != "", list(map(lambda x:x.strip(), texts))))
    newTexts = []
    assert len(texts) % 2 == 0, ValueError("split num is not double," + str(len(texts)) + "content:"+ lawText)
    i = 0
    while i < len(texts) - 1:
        newTexts.append(texts[i] + texts[i + 1])
        i += 2

    rfInputVector = []
    for i, text in enumerate(newTexts):
        inputText = ""
        if i == 0:
            inputText += 'S'
        else:
            inputText += newTexts[i-1]
        inputText += text

        if i == len(newTexts) - 1:
            inputText += 'E'
        else:
            inputText += newTexts[i+1]

        vector = processMultiLawText2id(inputText, dictionary)
        rfInputVector.append(vector)

    return outputTexts, rfInputVector














