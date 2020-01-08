#把与刑法133条匹配的，label为1的随机删除掉一半
import models.parameter as param
import random
import processLaw as psLaw
import re
import pickle


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
# buildDictQHJ()