import os
import json

#校正标注数据集
class DataCheck:
    def merge(self,sourceDir,targetFile):
        files = os.listdir(sourceDir)
        all_data = {}
        flag = True
        for file in files:
            file_path = os.path.join(sourceDir, file)
            fr = open(file_path, 'r', encoding='utf-8')
            data = json.load(fr)
            if flag:
                for k, v in data.items():
                    all_data[k] = v
                flag = False
            else:
                all_data.update(data)

        print('清除前共有{0}篇文书'.format(len(all_data.keys())))
        #清除没有标注数据的文书
        count = 0
        to_cleaned = []
        for k,v in all_data.items():
            if len(v) == 0:
                to_cleaned.append(k)
                count += 1
        while len(to_cleaned) > 0:
            k = to_cleaned.pop()
            all_data.pop(k)

        print('清除后共有{0}篇文书,共清除{1}篇文书'.format(len(all_data.keys()), count))

        fw = open(targetFile, 'w', encoding='utf-8')
        json.dump(all_data, fw)

    def checkIfLawExist(self,sourceFile, law):
        fr = open(sourceFile, 'r', encoding='utf-8')
        all_data = json.load(fr)
        for k, v in all_data.items():
            for v_ in v:
                if str(v_[1]).startswith(law):
                    print('Exist!')
                    return
        print('No exist!')
        return

    def CheckSampleNumberOfLaw(self,sourceFile, law):
        count = 0
        fr = open(sourceFile, 'r', encoding='utf-8')
        all_data = json.load(fr)
        for k, v in all_data.items():
            for v_ in v:
                if str(v_[1]).startswith(law):
                    count += 1
        return count

    def CheckSampleNumberOfDataSet(self,sourceFile):
        count = 0
        fr = open(sourceFile, 'r', encoding='utf-8')
        all_data = json.load(fr)
        for k, v in all_data.items():
            count += len(v)
        return count

    def rmLaw(self,sourceFile, law, targetFile):
        fr = open(sourceFile, 'r', encoding='utf-8')
        all_data = json.load(fr)
        for k , v in all_data.items():
            v = list(filter(lambda x:not x[1].startswith(law), v))
            all_data[k] = v
        fr.close()

        fw = open(targetFile, 'w', encoding='utf-8')
        json.dump(all_data, fw)

    def reLabeledLaw(self,sourceFile, law, targetFile):
        fr = open(sourceFile, 'r', encoding='utf-8')
        all_data = json.load(fr)
        count = 0
        for k, v in all_data.items():
            new_v = []
            for v_ in v:
                fact, ft, init_label = v_[0], v_[1], v_[2]
                if str(v_[1]).startswith(law):
                    #重新标注
                    print("事实:\n{0}\n法条:\n{1}".format(fact, ft))
                    while 1:
                        print('请标注{0}.........'.format(count))
                        label = input()
                        if label in ['0', '1', '2']:
                            new_v.append([fact, ft, label])
                            count += 1
                            break
                else:
                    new_v.append([fact, ft, init_label])
            all_data[k] = new_v
        fr.close()

        count = 0
        for k, v in all_data.items():
            count += len(v)
        print('标注后的样本数为'.format(count))

        fw = open(targetFile, 'w', encoding='utf-8')
        json.dump(all_data, fw)
        fw.close()

if __name__ == "__main__":
    DC = DataCheck()

    # sourceDir = '../resource/原始数据/故意伤害罪标注数据'
    # targetFile = sourceDir + '/合并.json'
    # DC.merge(sourceDir,targetFile)

    # sourceDir = '../resource/原始数据/故意伤害罪标注数据'
    # sourceFile = sourceDir + '/合并-new-v1.json'
    # laws = ['中华人民共和国刑事诉讼法(2013)第一百九十五条','中华人民共和国刑法(2015)第六十一条'] #被删除的法条
    # laws = ['中华人民共和国侵权责任法(2010)第十六条'] #需要重新标注的法条
    # targetFile = sourceDir + '/合并-new-v1.json'
    # for law in laws:
    #     sourceFile = targetFile
    #     targetFile = sourceDir + '/合并-new-v2.json'
    #     DC.reLabeledLaw(sourceFile,law, targetFile=targetFile)
    # sourceFi1le = sourceDir + '/合并-new-v2.json'
    # print(DC.CheckSampleNumberOfDataSet(sourceFile))

