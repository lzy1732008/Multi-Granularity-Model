from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
from processLaw import buildDataSetForRF
from sklearn.svm import SVC
import json
import pickle

# X,Y = buildDataSetForRF(train=True,val=True,test=False)
# X,Y = np.array(X), np.array(Y)
# rf = RandomForestClassifier(n_estimators=100)
# rf.fit(X,Y)
# with open('../result/model/RandomForest/.pkl','wb') as fw:
#      pickle.dump(rf,fw)

with open('../result/model/RandomForest/rf_rm2json-dict30bool-rules-v2.pkl', 'rb') as fr:
    rf = pickle.load(fr)

# testX, testY = buildDataSetForRF(train=False,val=False,test=True)
# testX, testY = np.array(testX), np.array(testY)
# pred_y = rf.predict(testX)
# print(metrics.classification_report(testY,pred_y,digits=4))
# print(metrics.accuracy_score(testY,pred_y))
# fr = open('../resource/lawDataSet_rm2_rmContext.json', 'r', encoding='utf-8')
# env = json.load(fr)
# count = 0
# for py, sample in zip(pred_y,env['testSet']):
#     input, target_y = sample[0], int(sample[1])
#     if target_y == py:
#         count += 1
#     else:
#         print("sample:{0}, label:{1}, predict:{2}".format(input,target_y,py))
# print(count/len(pred_y))





# X,Y = buildDataSetForRF(train=True,val=True,test=False)
# X,Y = np.array(X), np.array(Y)
# testX, testY = buildDataSetForRF(train=False,val=False,test=True)
# testX, testY = np.array(testX), np.array(testY)
# svm = SVC(kernel='linear')
# svm.fit(X,Y)
# pred_y = svm.predict(testX)
# print(metrics.classification_report(testY,pred_y,digits=4))
# print(metrics.accuracy_score(testY,pred_y))



#随机森林预测
import processLaw as psLaw
import  re
ft = '根据刑法第六十七条第一款的规定，犯罪以后自动投案，如实供述自己的罪行的，是自首。（一）自动投案，是指犯罪事实或者犯罪嫌疑人未被司法机关发觉，或者虽被发觉，但犯罪嫌疑人尚未受到讯问、未被采取强制措施时，主动、直接向公安机关、人民检察院或者人民法院投案。犯罪嫌疑人向其所在单位、城乡基层组织或者其他有关负责人员投案的；犯罪嫌疑人因病、伤或者为了减轻犯罪后果，委托他人先代为投案，或者先以信电投案的；罪行尚未被司法机关发觉，仅因形迹可疑，被有关组织或者司法机关盘问、教育后，主动交代自己的罪行的；犯罪后逃跑，在被通缉、追捕过程中，主动投案的；经查实确已准备去投案，或者正在投案途中，被公安机关捕获的，应当视为自动投案。并非出于犯罪嫌疑人主动，而是经亲友规劝、陪同投案的；公安机关通知犯罪嫌疑人的亲友，或者亲友主动报案后，将犯罪嫌疑人送去投案的，也应当视为自动投案。犯罪嫌疑人自动投案后又逃跑的，不能认定为自首。（二）如实供述自己的罪行，是指犯罪嫌疑人自动投案后，如实交代自己的主要犯罪事实。犯有数罪的犯罪嫌疑人仅如实供述所犯数罪中部分犯罪的，只对如实供述部分犯罪的行为，认定为自首。共同犯罪案件中的犯罪嫌疑人，除如实供述自己的罪行，还应当供述所知的同案犯，主犯则应当供述所知其他同案犯的共同犯罪事实，才能认定为自首。犯罪嫌疑人自动投案并如实供述自己的罪行后又翻供的，不能认定为自首；但在一审判决前又能如实供述的，应当认定为自首。'
outputTexts, rfInputVector  = psLaw.processLawForRf(ft)
labels = rf.predict(rfInputVector)
content_split = re.split(r"[，；。：]",outputTexts)
content_split = list(filter(lambda x: x != "", list(map(lambda x: x.strip(), content_split))))
for law, label in zip(content_split,labels):
    if label == 0:
        print(law)



