import json
import numpy as np
import math

# def softmax(input):
#     exp_input = [math.exp(x) for x in input]
#     exp_sum = sum(exp_input)
#     return [float(x)/exp_sum for x in exp_input]
#
fr1 = open('resource/预测结果分析/MGCQ_16_predictAna.json','r',encoding='utf-8')
fr2 = open('resource/预测结果分析/MGCQ_16_predictAna.json','r',encoding='utf-8')
model1_result = json.load(fr1)
model2_result = json.load(fr2)

#分析模型2 对比 模型1中各类概率增大
# enhanced_1 = []
# enhanced_0 = []

#原本预测错误的被预测正确的
#原本预测正确的被预测错误的
wrong_to_right = []
right_to_wrong = []
predict_wrong = []
for k,v in model1_result.items():
    for k_1, v_1 in v.items():
        v_1_2 = list(map(float,v_1[2]))

        # if k == '中华人民共和国刑法(2015)第一百三十三条:违反交通运输管理法规，因而发生重大事故，致人重伤、死亡或者使公私财产遭受重大损失的，处三年以下有期徒刑或者拘役；交通运输肇事后逃逸或者有其他特别恶劣情节的，处三年以上七年以下有期徒刑；因逃逸致人死亡的，处七年以上有期徒刑。':
        #     if v_1[0] == 0:
        #         print(k_1,v_1)
        if v_1[0] != v_1[1]:
            predict_wrong.append([k,k_1,v_1])
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

with open('resource/预测结果分析/模型16预测错误-基于法条排序.txt','w',encoding='utf-8') as fw:
    fw.write('wrong to right.....\n'+'\n'.join(list(map(str,predict_wrong)))+'\n')
    # fw.write('right to wrong.....\n' + '\n'.join(list(map(str, right_to_wrong))))


# ft = '中华人民共和国刑法(2015)第一百三十三条:违反交通运输管理法规，因而发生重大事故，致人重伤、死亡或者使公私财产遭受重大损失的，处三年以下有期徒刑或者拘役；交通运输肇事后逃逸或者有其他特别恶劣情节的，处三年以上七年以下有期徒刑；因逃逸致人死亡的，处七年以上有期徒刑。'
# import models.parameter as param
# lines = open('resource/test-init-完整.txt','r',encoding='utf-8').read().split('\n')
# label_0 = []
# count = 0
# for line in lines:
#     line = line.strip()
#     if line != '':
#         items = line.split('|')
#         if items[2] == ft:
#             if items[-1] == '0':
#                 label_0.append(items[1])
#             else:
#                 count += 1
# print(len(label_0))
# print(count)
# print('\n'.join(label_0))



