#analyse the relationship between CoA and law in criminal cases

from util.ws_fun import getFTList, getFTfromQW
import random
import os
import xlrd
import xlwt

def law_in_each_coa(source_dir_list, law):
    #shuffle docs
    files_path = []
    for source_dir in source_dir_list:
        files_ = os.listdir(source_dir)
        files_path += [os.path.join(source_dir,file) for file in files_]

    indexs = random.sample([i for i in range(len(files_path))], 10000)

    count = 0
    for i in indexs:
        if not str(files_path[i]).endswith('.xml'): continue
        path = files_path[i]
        ftlist = getFTfromQW(path)
        for ft in ftlist:
            if str(ft).count(law)>0:
                count += 1
                break
    print(count)

def laws_in_coa(source_dir_list):
    files_path = []
    for source_dir in source_dir_list:
        files_ = os.listdir(source_dir)
        files_path += [os.path.join(source_dir, file) for file in files_]

    indexs = random.sample([i for i in range(len(files_path))], 10000)

    law_dict = {}
    for i in indexs:
        if not str(files_path[i]).endswith('.xml'): continue
        path = files_path[i]
        ftmclist= getFTfromQW(path)
        for ft in ftmclist:
            if ft not in law_dict.keys():law_dict[ft] = 0
            law_dict[ft] += 1

    excel = xlwt.Workbook()
    sheet = excel.add_sheet('law_distribution')
    row = 0

    laws = sorted(law_dict.items(),key=lambda x:x[1],reverse=True)
    for ft, n in laws:
        print(ft)
        sheet.write(row, 1, ft)
        sheet.write(row, 2, n)
        row += 1

    excel.save('../resource/ana/法条分布分析_毒品罪.xls')



if __name__ == '__main__':
    source_dir_1 = '/Users/wenny/nju/task/文书整理/走私、贩卖、运输、制造毒品罪2015/2015'
    # source_dir_2 = '/Users/wenny/nju/task/文书整理/信用卡诈骗/2014填充'
    law = '《中华人民共和国刑法》第七十二条'
    # law_in_each_coa([source_dir_1],law)

    laws_in_coa([source_dir_1])


