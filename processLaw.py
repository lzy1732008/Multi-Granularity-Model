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
    sourceLaw = 'resource/ft_nr_QHSplit.txt'
    labeledLaw = 'resource/ftSplitCheck.txt'
    targetLaw = 'resource/ft_labeled.json'
    fr1 = open(sourceLaw,'r',encoding='utf-8')
    fr2 = open(labeledLaw,'r',encoding='utf-8')
    fw = open(targetLaw,'w',encoding='utf-8')
    source1_dict = {}
    source2_dict = {}
    #获取第一个源文件的内容
    lines = fr1.readlines()
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        array = line.split('|')
        ft = array[1].strip()
        content = array[2].strip()
        if ft == "" or content == "":
            continue

        source1_dict[ft] = content

    lines = fr2.read().split('法条:')
    for line in lines:
        array = line.split('\n')
        labeledDatas = []
        for c in array[1:]:
            c = c.strip()
            if c == "": continue
            labeled = c.split('\t')
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
            assert str(content[endIndex]) in punctuation, ValueError("End char isn't punctuation!")
            contents.append([content[startIndex:endIndex+1],subitem[1]])
            content = content[endIndex+1:]
        output_dict[ft] = contents
    json.dump(output_dict,fw)

import random
import numpy as np

def buildDataSet(window = 1):
    jieba.load_userdict('resource/dict.txt')
    fw = open('resource/lawDataSet.json','w',encoding='utf-8')
    dataset = {}
    fr = open('resource/ft_labeled.json','r',encoding='utf-8')
    allft = json.load(fr).items()
    alldata = []
    for ft,labeledcontents in allft:
        for i, c in enumerate(labeledcontents):
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
    dataset["trainSet"] = trainSet
    dataset["valSet"] = valSet
    dataset["testSet"] = testSet
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
        wordEmbedding[nw] = newWordsEmbedding[i].tolist()

    w_word = open(param.BaseConfig.w2vModel_ex, 'w', encoding='utf-8')
    json.dump(wordEmbedding,w_word)

# extendWordVocabs()

















