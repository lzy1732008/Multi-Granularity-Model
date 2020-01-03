#-*-coding:utf-8
import json
import numpy as np
import math

# def softmax(input):
#     exp_input = [math.exp(x) for x in input]
#     exp_sum = sum(exp_input)
#     return [float(x)/exp_sum for x in exp_input]

fr1 = open('resource/预测结果分析/MGCQ_16_predictAna-qj-v4.json','r',encoding='utf-8')
fr2 = open('resource/预测结果分析/MGCQ_23-qj.json','r',encoding='utf-8')
model1_result = json.load(fr1)
model2_result = json.load(fr2)

#分析模型2 对比 模型1中各类概率增大
# enhanced_1 = []
# enhanced_0 = []h

#原本预测错误的被预测正确的
#原本预测正确的被预测错误的
from sklearn import metrics
wrong_to_right = []
right_to_wrong = []
predict_wrong = []
count_right = 0
count = 0
for k,v in model2_result.items():
    true_y = []
    pred_y = []
    for k_1, v_1 in v.items():
        if v_1[1] != v_1[0]:
            print(k,k_1,v_1)

        # if v_1[1] == v_1[0] == 1:
        #     print(k)
    #==============计算每个法条的1的准确率和召回率=======
        # true_y.append(v_1[1])
        # pred_y.append(v_1[0])
    # print('法条:'+k+'......')
    # print("Precision, Recall and F1-Score...")
    # print(metrics.classification_report(np.array(true_y), np.array(pred_y), digits=4))  # 直接计算准确率，召回率和f值
    #
    # # 混淆矩阵
    # print("Confusion Matrix...")
    # cm = metrics.confusion_matrix(np.array(true_y), np.array(pred_y))
    # print(cm)
    #
    # true_y = []
    # pred_y = []
    # v_model2 = model2_result[k]
    # for k_2,v_2 in v_model2.items():
    #     true_y.append(v_2[1])
    #     pred_y.append(v_2[0])
    # print('模型16......')
    # print("Precision, Recall and F1-Score...")
    # print(metrics.classification_report(np.array(true_y), np.array(pred_y), digits=4))  # 直接计算准确率，召回率和f值
    #
    # # 混淆矩阵
    # print("Confusion Matrix...")
    # cm = metrics.confusion_matrix(np.array(true_y), np.array(pred_y))
    # print(cm)



        # v_1_2 = list(map(float,v_1[2]))
        # ft = '最高人民法院关于审理交通肇事刑事案件具体应用法律若干问题的解释(2000)第二条:交通肇事具有下列情形之一的，处三年以下有期徒刑或者拘役：（一）死亡一人或者重伤三人以上，负事故全部或者主要责任的；（二）死亡三人以上，负事故同等责任的；（三）造成公共财产或者他人财产直接损失，负事故全部或者主要责任，无能力赔偿数额在三十万元以上的。交通肇事致一人以上重伤，负事故全部或者主要责任，并具有下列情形之一的，以交通肇事罪定罪处罚：（一）酒后、吸食毒品后驾驶机动车辆的；（二）无驾驶资格驾驶机动车辆的；（三）明知是安全装置不全或者安全机件失灵的机动车辆而驾驶的；（四）明知是无牌证或者已报废的机动车辆而驾驶的；（五）严重超载驾驶的；（六）为逃避法律追究逃离事故现场的。'
        # ft = '最高人民法院关于处理自首和立功具体应用法律若干问题的解释(1998)第三条:根据刑法第六十七条第一款的规定，对于自首的犯罪分子，可以从轻或者减轻处罚；对于犯罪较轻的，可以免除处罚。具体确定从轻、减轻还是免除处罚，应当根据犯罪轻重，并考虑自首的具体情节。'
        # ft = '中华人民共和国刑法(2015)第一百三十三条:违反交通运输管理法规，因而发生重大事故，致人重伤、死亡或者使公私财产遭受重大损失的，处三年以下有期徒刑或者拘役；交通运输肇事后逃逸或者有其他特别恶劣情节的，处三年以上七年以下有期徒刑；因逃逸致人死亡的，处七年以上有期徒刑。'
        # if k == ft and v_1[0] == v_1[1] == 0:
        #     print(k_1)
        #     count_right += 1
        # if k == ft:
        #     count += 1
        # if v_1[0] != v_1[1]:
        #     predict_wrong.append([k,k_1,v_1])
        # model2_v_1 = list(map(float,model2_result[k][k_1][2]))
        # softmax_1 = softmax(v_1_2)
        # softmax_2 = softmax(model2_v_1)
        # dis = np.array(softmax_2) - np.array(softmax_1)
        # if v_1[1] == v_1[0] and v_1[1] != model2_result[k][k_1][0]:
        #     right_to_wrong.append([k,k_1,v_1])
        # if v_1[1] != v_1[0] and v_1[1] == model2_result[k][k_1][0]:
        #     wrong_to_right.append([k,k_1,v_1])
        # if v_1[1] == 0:
        #     if dis[0] > 0:
        #         enhanced_0.append([k,k_1,v_1,dis[0]])
        # else:
        #     if dis[1] > 0:
        #         enhanced_1.append([k,k_1,v_1,dis[1]])
# print('0的准确率:'+str(count_right/count))
# with open('resource/预测结果分析/模型16预测错误-基于法条排序.txt','w',encoding='utf-8') as fw:
#     fw.write('wrong to right.....\n'+'\n'.join(list(map(str,predict_wrong)))+'\n')
    # fw.write('right to wrong.....\n' + '\n'.join(list(map(str, right_to_wrong))))


# ft = '中华人民共和国刑法(2015)第一百三十三条:违反交通运输管理法规，因而发生重大事故，致人重伤、死亡或者使公私财产遭受重大损失的，处三年以下有期徒刑或者拘役；交通运输肇事后逃逸或者有其他特别恶劣情节的，处三年以上七年以下有期徒刑；因逃逸致人死亡的，处七年以上有期徒刑。'
# ft = '中华人民共和国刑法(2015)第七十三条:拘役的缓刑考验期限为原判刑期以上一年以下，但是不能少于二个月。有期徒刑的缓刑考验期限为原判刑期以上五年以下，但是不能少于一年。缓刑考验期限，从判决确定之日起计算。'
# ft = '中华人民共和国刑法(2015)第七十二条:对于被判处拘役、三年以下有期徒刑的犯罪分子，同时符合下列条件的，可以宣告缓刑，对其中不满十八周岁的人、怀孕的妇女和已满七十五周岁的人，应当宣告缓刑：（一）犯罪情节较轻；（二）有悔罪表现；（三）没有再犯罪的危险；（四）宣告缓刑对所居住社区没有重大不良影响。宣告缓刑，可以根据犯罪情况，同时禁止犯罪分子在缓刑考验期限内从事特定活动，进入特定区域、场所，接触特定的人。被宣告缓刑的犯罪分子，如果被判处附加刑，附加刑仍须执行。'
# ft= '最高人民法院关于处理自首和立功具体应用法律若干问题的解释(1998)第一条:根据刑法第六十七条第一款的规定，犯罪以后自动投案，如实供述自己的罪行的，是自首。（一）自动投案，是指犯罪事实或者犯罪嫌疑人未被司法机关发觉，或者虽被发觉，但犯罪嫌疑人尚未受到讯问、未被采取强制措施时，主动、直接向公安机关、人民检察院或者人民法院投案。犯罪嫌疑人向其所在单位、城乡基层组织或者其他有关负责人员投案的；犯罪嫌疑人因病、伤或者为了减轻犯罪后果，委托他人先代为投案，或者先以信电投案的；罪行尚未被司法机关发觉，仅因形迹可疑，被有关组织或者司法机关盘问、教育后，主动交代自己的罪行的；犯罪后逃跑，在被通缉、追捕过程中，主动投案的；经查实确已准备去投案，或者正在投案途中，被公安机关捕获的，应当视为自动投案。并非出于犯罪嫌疑人主动，而是经亲友规劝、陪同投案的；公安机关通知犯罪嫌疑人的亲友，或者亲友主动报案后，将犯罪嫌疑人送去投案的，也应当视为自动投案。犯罪嫌疑人自动投案后又逃跑的，不能认定为自首。（二）如实供述自己的罪行，是指犯罪嫌疑人自动投案后，如实交代自己的主要犯罪事实。犯有数罪的犯罪嫌疑人仅如实供述所犯数罪中部分犯罪的，只对如实供述部分犯罪的行为，认定为自首。共同犯罪案件中的犯罪嫌疑人，除如实供述自己的罪行，还应当供述所知的同案犯，主犯则应当供述所知其他同案犯的共同犯罪事实，才能认定为自首。犯罪嫌疑人自动投案并如实供述自己的罪行后又翻供的，不能认定为自首；但在一审判决前又能如实供述的，应当认定为自首。'
# ft = '中华人民共和国民法通则(2009)第一百一十九条:侵害公民身体造成伤害的，应当赔偿医疗费、因误工减少的收入、残废者生活补助费等费用；造成死亡的，并应当支付丧葬费、死者生前扶养的人必要的生活费等费用。'
# ft = '最高人民法院关于审理交通肇事刑事案件具体应用法律若干问题的解释(2000)第二条:交通肇事具有下列情形之一的，处三年以下有期徒刑或者拘役：（一）死亡一人或者重伤三人以上，负事故全部或者主要责任的；（二）死亡三人以上，负事故同等责任的；（三）造成公共财产或者他人财产直接损失，负事故全部或者主要责任，无能力赔偿数额在三十万元以上的。交通肇事致一人以上重伤，负事故全部或者主要责任，并具有下列情形之一的，以交通肇事罪定罪处罚：（一）酒后、吸食毒品后驾驶机动车辆的；（二）无驾驶资格驾驶机动车辆的；（三）明知是安全装置不全或者安全机件失灵的机动车辆而驾驶的；（四）明知是无牌证或者已报废的机动车辆而驾驶的；（五）严重超载驾驶的；（六）为逃避法律追究逃离事故现场的。'
#获取每个法条在训练集合中正负样本的比===========
# import models.parameter as param
# lines = open('resource/train-init.txt','r',encoding='utf-8').read().split('\n')
# label_1 = []
# count = 0
# law_dict = {}
# for line in lines:
#     line = line.strip()
#     if line != '':
#         items = line.split('|')
#         if items[2] not in law_dict.keys():
#             law_dict[items[2]] = [0,0]
#         if items[-1] == '0':
#             law_dict[items[2]][0] += 1
#         else:
#             law_dict[items[2]][1] += 1
# for k,v in law_dict.items():
#     print(k,v)
# print(len(label_1), count - len(label_1))

# lines = open('resource/train-init.txt','r',encoding='utf-8').read().split('\n')
# new_lines = []
# count = 0
# law_dict = {}
# for line in lines:
#     line = line.strip()
#     if line != '':
#         items = line.split('|')
#         if items[2] == ft and items[-1] == '0':
#             new_lines.append(line)
#             count += 1
#             if count >= 100:
#                 break
# open('resource/test-init-test.txt','a',encoding='utf-8').write('\n'.join(new_lines))





