from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
from processLaw import buildDataSetForRF
from sklearn.svm import SVC
import json
import pickle

X,Y = buildDataSetForRF(train=True,val=True,test=False)
X,Y = np.array(X), np.array(Y)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X,Y)
with open('../result/model/RandomForest/rf_rm2json-dict30bool-rules-punction.pkl','wb') as fw:
     pickle.dump(rf,fw)

with open('../result/model/RandomForest/rf_rm2json-dict30bool-rules-punction.pkl', 'rb') as fr:
    rf = pickle.load(fr)

testX, testY = buildDataSetForRF(train=False,val=False,test=True)
testX, testY = np.array(testX), np.array(testY)
pred_y = rf.predict(testX)
print(metrics.classification_report(testY,pred_y,digits=4))
print(metrics.accuracy_score(testY,pred_y))
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




