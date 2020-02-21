#把与刑法133条匹配的，label为1的随机删除掉一半
import models.parameter as param
import random
import processLaw as psLaw
import re
import pickle
import os
import numpy as np
import shutil
import time
from sklearn import metrics
import gensim
import json



from util.ws_fun import getFTList, getRDSS, getZKSS, getQWChildContent
from util.evaluate import wsevaluate

#输入：词向量训练的语料库以及词向量模型
#输出：词向量字典，文件名word_embedding.json
def load_models(model_path):
    return gensim.models.Word2Vec.load(model_path)

def vector(v,model):
    try:
        return model[v]
    except:
        return [0]*128

def buildWordEmbeddingFile():
    source_dir = '../resource/故意伤害罪/词向量源文件'
    fw = open(os.path.join(source_dir, 'word_embedding.json'), 'w', encoding='utf-8')
    fr = open(os.path.join(source_dir, 'corpus.txt'), 'r', encoding='utf-8')
    model_path = os.path.join(source_dir, 'w2v_size128.model')
    w2v_model = load_models(model_path)

    corpus = fr.read().split('\n')
    words_embedding = {}

    for line in corpus:
        line_words = line.split()
        for word in line_words:
            word = word.strip()
            if word!= "" and word not in words_embedding.keys():
                words_embedding[word] = '\t'.join(list(map(str, vector(word, w2v_model))))

    words_embedding['<UNK>'] = '\t'.join(['0' for _ in range(param.BaseConfig.word_dimension)])
    json.dump(words_embedding, fw)

# buildWordEmbeddingFile()

#建立训练集、测试集、验证集
#输入：标注好的数据集:.json,数据格式为{'xml文件名': [[fact,law,label],[fact,law,label]]}
#输出：train.txt,val.txt, test.txt

def SplitDataSet(sourcePath,targetDir):
    fr = open(sourcePath,'r',encoding='utf-8')
    all_data_set = json.load(fr)


    #随机选择训练集：验证集：测试集=90%：5%:5%
    all_keys = list(all_data_set.keys())
    test_num, val_num = int(0.05 * len(all_keys)), int(0.05 * len(all_keys))
    test_index = random.sample([i for i in range(len(all_keys))], test_num)
    remained_keys = list(map(lambda y: all_keys[y], list(filter(lambda x: x not in test_index, [i for i in range(len(all_keys))]))))
    val_index = random.sample([i for i in range(len(remained_keys))], val_num)
    train_index = list(filter(lambda x: x not in val_index, [i for i in range(len(remained_keys))]))
    print('随机得到的训练集：{0},验证集:{1},测试集:{2}'.format(len(train_index), len(val_index), len(test_index)))

    train_set, val_set, test_set = set(train_index), set(val_index), set(test_index)
    assert len(train_set & val_set) == 0, ValueError("随机生成有重复元素")

    #根据随机生成的下标建立每个数据集的init.txt文件
    fw_train = open(targetDir+'/train-init.txt','w',encoding='utf-8')
    fw_test = open(targetDir + '/test-init.txt','w',encoding='utf-8')
    fw_val = open(targetDir + '/val-init.txt', 'w', encoding='utf-8')

    def createFile(all_data_set, selected_file, fw):
        train_output = []
        for file in selected_file:
            key_samples = all_data_set[file]
            for s in key_samples:
                #去除掉事实和法条里面的'\n'字符
                fact = str(s[0]).replace('\n','')
                law = str(s[1]).replace('\n','')
                line = '|'.join([file, fact, law, str(s[2])])
                train_output.append(line)
        print('样本个数:{0}'.format(len(train_output)))
        fw.write('\n'.join(train_output))

    # 首先建立训练集
    train_files = list(map(lambda x: remained_keys[x], train_index))
    test_files = list(map(lambda x: all_keys[x], test_index))
    val_files = list(map(lambda x: remained_keys[x], val_index))

    createFile(all_data_set, train_files, fw_train)
    createFile(all_data_set, test_files, fw_test)
    createFile(all_data_set, val_files, fw_val)



def rmLabel1InLaw():
    ft = '中华人民共和国刑法(2015)第一百三十三条:违反交通运输管理法规，因而发生重大事故，致人重伤、死亡或者使公私财产遭受重大损失的，处三年以下有期徒刑或者拘役；交通运输肇事后逃逸或者有其他特别恶劣情节的，处三年以上七年以下有期徒刑；因逃逸致人死亡的，处七年以上有期徒刑。'
    lines = open('../resource/val-init.txt','r',encoding='utf-8').read().split('\n')
    newlines = []
    for line in lines:
        line = line.strip()
        if line != "":
            items = line.split('|')
            if items[2] == ft and items[-1] == '1':
                pw = random.random()
                if pw > 0.5:
                #save this sample
                   newlines.append(line)
            else:
                newlines.append(line)


    open('../resource/val-删除1.txt','w',encoding='utf-8').write('\n'.join(newlines))

# rmLabel1InLaw()

#建立一个数据集，里面的格式是文件名|事实|法条前件|label
def buildCorpusWithQJ():
    with open('../result/model/RandomForest/rf_rm2json-dict30bool-rules-v2.pkl', 'rb') as fr:
        rfModel = pickle.load(fr)
    lines = open('../resource/test-init-alter-4.txt','r',encoding='utf-8').read().split('\n')
    newlines = []
    for line in lines:
        line = line.strip()
        if line != "":
            items = line.split('|')
            law_units = items[2].split(':')
            law_name = law_units[0]
            law_content = items[2][len(law_name) + 1:]
            law_content, law_input_vector = psLaw.processLawForRf(law_content)
            # 接下来预测每句话的label,并将其映射到每个词上
            law_labels = rfModel.predict(law_input_vector)
            content_split = re.split(r"[，；。：]", law_content)
            content_split = list(filter(lambda x: x != "", list(map(lambda x: x.strip(), content_split))))
            law_qj = []
            for label, law_split in zip(law_labels, content_split):
                if label == 0:
                    law_qj.append(law_split)
            newlines.append('|'.join([items[0],items[1],'。'.join(law_qj),items[-1]]))
    open('../resource/test-qj-alter-4.txt','w',encoding='utf-8').write('\n'.join(newlines))

# buildCorpusWithQJ()

#统计每个法条的平均前件个数
def countPreNum():
    laws = {}
    lines = open('../resource/train-init.txt', 'r', encoding='utf-8').read().split('\n')
    for line in lines:
        line = line.strip()
        if line != "":
            items = line.split('|')
            if items[2] not in laws.keys():
                laws[items[2]] = len(items[2].split('。'))
            else:
                continue

    values = laws.values()
    print(values)


# countPreNum()

#数据集扩增：通过对法条增加两条乱序
def shuffleLaw(times_default, shuffle_num_default):
    lines = open('../resource/train-qj.txt', 'r', encoding='utf-8').read().split('\n')
    newlines = []
    count = 0

    #对每个法条都增加几个乱序，存储在字典里面

    for line in lines:
        line = line.strip()
        if line != "":
            count += 1
            items = line.split('|')
            law_content_split = items[2].split('。')
            law_content_split = list(filter(lambda x:x.strip() != "", law_content_split))
            #乱序
            shuffle_num = min(shuffle_num_default, int(len(law_content_split)/2))
            times = times_default
            if len(law_content_split) <= 2:
                times = 1
            # print("the length of law split:" + str(len(law_content_split)))
            for i in range(times):
                new_law_content_split = list(law_content_split)
                for i in range(shuffle_num):
                    num = random.randint(1, len(law_content_split) - 1)
                    new_law_content_split[num - 1], new_law_content_split[num] = new_law_content_split[num], \
                                                                                 new_law_content_split[num - 1]
                newlines.append('|'.join([items[0], items[1], '。'.join(new_law_content_split), items[-1]]))
            # newlines.append(line)
    print('总共样本数由{0}变为{1}'.format(count, len(newlines)))
    open('../resource/train-qj-augment-rminit.txt', 'w', encoding='utf-8').write('\n'.join(newlines))

# shuffleLaw(times_default=2,shuffle_num_default=2)




import json
#建立一个词典，里面存放每个法条的前后件信息
def buildDictQHJ():
    with open('../result/model/RandomForest/rf_rm2json-dict30bool-rules-v2.pkl', 'rb') as fr:
        rfModel = pickle.load(fr)
    lines = open('../resource/train-init.txt','r',encoding='utf-8').read().split('\n')
    lines += open('../resource/test-init.txt','r',encoding='utf-8').read().split('\n')
    lines += open('../resource/val-init.txt', 'r', encoding='utf-8').read().split('\n')
    law_dict = {}
    for line in lines:
        line = line.strip()
        if line != "":
            items = line.split('|')
            law_units = items[2].split(':')
            law_name = law_units[0]
            if law_name in law_dict.keys(): continue

            law_content = items[2][len(law_name) + 1:]
            law_content, law_input_vector = psLaw.processLawForRf(law_content)
            # 接下来预测每句话的label,并将其映射到每个词上
            law_labels = rfModel.predict(law_input_vector)
            content_split = re.split(r"[，；。：]", law_content)
            content_split = list(filter(lambda x: x != "", list(map(lambda x: x.strip(), content_split))))
            law_qj = []
            law_hj = []
            for label, law_split in zip(law_labels, content_split):
                if label == 0:
                    law_qj.append(law_split)
                else:
                    law_hj.append(law_split)
            law_dict[law_name]={'qj':'。'.join(law_qj),'hj':'。'.join(law_hj),'whole':items[2][len(law_name) + 1:]}
    with open('../resource/law_qhj_dict.json','w',encoding='utf-8') as fw:
        json.dump(law_dict, fw)

#统计交通肇事文书中所有的法条，及其正文

def getAllFtInCase():
    dir_path = '/Users/wenny/nju/task/LawDocumentAna/2014filled/2014'
    files = os.listdir(dir_path)
    outputdir = '../resource/alljtzsft.json'
    law_dict = {}
    for file in files:
        if not file.endswith('.xml'): continue
        wspath = os.path.join(dir_path, file)
        ftmcls, ftnrls = getFTList(wspath)
        for mc, nr in zip(ftmcls,ftnrls):
            if mc not in law_dict.keys() and nr.strip() != "NOT FOUND":
                law_dict[mc] = nr
            if str(nr).count("FOUND") > 0:
                print("Here!")

    fw = open(outputdir,'w',encoding='utf-8')
    json.dump(law_dict,fw)


#对故意伤害罪案由进行标注========================================================
#随机选择500篇文书
def chooseWsRandom(dir_path, target_path):
    files = os.listdir(dir_path)
    files = list(filter(lambda x:str(x).endswith('.xml'), files))
    chose_file = []
    while len(chose_file) < 500:
        index = random.randint(0,len(files) - 1)
        if files[index] not in chose_file:
            chose_file.append(files[index])
    for file in chose_file:
        shutil.copy(src=os.path.join(dir_path, file), dst=target_path)

# dir_path = '/Users/wenny/nju/task/文书整理/故意伤害罪/2013填充'
# target_path = '../resource/原始数据/故意伤害罪'
# chooseWsRandom(dir_path,target_path)

def labelInferenceData(src_path, store_dir_path):
    store_file_name = 'jtzs_inference_labeleddata_0.json'
    store_path = os.path.join(store_dir_path,store_file_name)
    file_num = 0
    if os.path.exists(store_path):
        print("文件已存在！请输入新文件序号！")
        file_num = int(input())
        store_file_name = 'jtzs_inference_labeleddata_' + str(file_num) + '.json'
    store_path = os.path.join(store_dir_path, store_file_name)
    fw = open(store_path, 'w', encoding='utf-8')

    labeled_data_dict = {}
    start_time = time.time()

    files = os.listdir(src_path)
    total_label_count = 0
    for index in range(file_num, len(files)):
        file = files[index]
        print("处理到第{0}个文书,文书名为{1}.........".format(index, file))
        path = os.path.join(src_path, file)

        cpfxdl = getQWChildContent(path, 'CPFXGC')
        fact_str = getRDSS(path)
        fact_str += getZKSS(path)
        fact_split = list(set(list(filter(lambda x: x != "", list(map(lambda x:x.strip(), re.split(r'[。；]',fact_str)))))))
        ftnamelist, ftnrlist = getFTList(path)
        print("该文书共有{0}个待标注数据".format(len(fact_split) * len(ftnamelist)))
        labeled_count = 0
        labeled_sample = []
        for ftname, ftnr in zip(ftnamelist,ftnrlist):
            for fact in fact_split:
                print("事实:\n{0}\n法条:\n{1}".format(fact, ftname + ':' + ftnr))
                while 1:
                    print('请标注.........')
                    label = input()
                    if label == 'h':
                        print('裁判分析段落为:\n'+cpfxdl+'\n')
                        continue
                    if label == 's':
                        end_time = time.time()
                        print('正在停止本次标注........')
                        json.dump(labeled_data_dict, fw)
                        print('保存本次标注成功！')
                        print('本次共标注{0}条数据，耗时{1}s,标注到第{2}篇文书'.format(total_label_count, end_time - start_time, index))
                        print('退出程序！')
                        return

                    if label in ['0','1','2']:
                        labeled_count += 1
                        labeled_sample.append([fact, ftname + ':' + ftnr, int(label)])
                        break
        total_label_count += labeled_count
        labeled_data_dict[file] = labeled_sample
        print("该文书标注处理完毕！")

    end_time = time.time()
    print('正在停止本次标注........')
    json.dump(labeled_data_dict, fw)
    print('保存本次标注成功！')
    print('本次共标注{0}条数据，耗时{1}s'.format(total_label_count, end_time - start_time))
    print('退出程序！ ')
    return

# src_path = '../resource/原始数据/故意伤害罪文书'
# store_path = '../resource/原始数据/故意伤害罪标注数据'
# labelInferenceData(src_path, store_path)

# files = os.listdir(store_path)
# count = 0
# pos = 0
# neg = 0
# for file in files:
#     file_path = os.path.join(store_path, file)
#     fr = open(file_path,'r',encoding='utf-8')
#     labeled_data = json.load(fr)
#     for k , v in labeled_data.items():
#         for v_ in v:
#             if v_[-1] == 0:
#                 neg += 1
#             else:
#                 pos += 1
# print(neg,pos)
# print('总数为:',count)

# fr = open(os.path.join(store_path,'jtzs_inference_labeleddata_155.json'),'r',encoding='utf-8')
# labeled_data = json.load(fr)
# print(labeled_data['632683.xml'])


#统计学的方法
#以50%的概率随机猜测

#基于每个法条的正负例比例随机预测
def getRatio():
    ratio_dict = {}
    lines = open('../resource/ana/故意伤害罪_法条正负样本分布.txt','r',encoding='utf-8').read().split('\n')
    for line in lines:
        line = line.strip()
        if line != "":
            split = line.split(' [')
            assert len(split) == 2, ValueError("The number of line is wrong, "+ line)
            ratio = split[-1][:-1].split(', ')
            negtive = int(ratio[0])
            positive = int(ratio[1])
            total = negtive + positive
            neg_prob, pos_prob = 0, 0
            if total !=0 :
                neg_prob = negtive/ total
                pos_prob = positive/ total
            ratio_dict[split[0]] = [neg_prob, pos_prob]
    fw = open('../resource/ana/故意伤害罪_法条正负分布比例.json','w',encoding='utf-8')
    json.dump(ratio_dict, fw)

# getRatio()

#使用上一步得到的法条正负例分布比率进行预测
def statisticsPredict(targetFile):
    fr = open('../resource/ana/故意伤害罪_法条正负分布比例.json','r',encoding='utf-8')
    law_ratio = json.load(fr)

    lines = open(targetFile,'r',encoding='utf-8').read().split('\n')

    predict_y = []
    target_y = []
    for line in lines:
        line = line.strip()
        if line == "":continue
        split = line.split('|')
        law_content = split[2]
        # ratio = [0.7137,0.2862] #交通肇事罪
        ratio = [0.6806,0.3193] #故意伤害罪
        if law_content in law_ratio.keys():
           ratio = law_ratio[law_content]
        predict_y.append(getRandomSample(number_list=[0,1], pro_list=ratio))
        target_y.append(int(split[-1]))

    print(metrics.classification_report(target_y, predict_y, digits=4))  # 直接计算准确率，召回率和f值

    # 混淆矩阵
    # print("Confusion Matrix...")
    # cm = metrics.confusion_matrix(target_y, predict_y)
    # print(cm)
    return predict_y, target_y

def getRandomSample(number_list, pro_list):
    x = random.uniform(0, 1)
    # 累积概率
    cum_pro = 0.0
    # 将可迭代对象打包成元组列表
    for number, number_pro in zip(number_list, pro_list):
        cum_pro += number_pro
        if x < cum_pro:
            # 返回值
            return number

def getwslist():
    lines = open('../resource/gyshz_traindata/test-init.txt','r',encoding='utf-8').read().split('\n')
    namels = []
    for i in range(len(lines)):
        line = lines[i]
        if line.strip() == "":
            continue
        array = line.split('|')

        if len(array) < 4:
            continue

        namels.append(array[0])
    return namels

if __name__ == "__main__":
    # sourcePath = '../resource/原始标注数据/故意伤害罪标注数据/合并-new-v2.json'
    # targetDir = '../resource/故意伤害罪训练数据集'
    # SplitDataSet(sourcePath, targetDir)

#使用统计方法预测
    # getRatio()
    # targetFile = '../resource/gyshz_traindata/test-init.txt'
    # predict_y, target_y = statisticsPredict(targetFile)
    # wslist = getwslist()
    # assert len(predict_y) == len(wslist), ValueError("The number of ws is not equal to the model predict")
    # wsevaluate(predict_y, target_y,wslist)





