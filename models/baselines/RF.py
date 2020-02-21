from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from sklearn import metrics
from sklearn.svm import SVC

from util.evaluate import wsevaluate
from util.file_fun import getwslist

def vector_text(input):
    features = open('../../resource/gyshz_feature_set.txt','r',encoding='utf-8').read().split('\n')
    vector = []
    for w in features:
        vector.append(str(input).count(w))

    return vector

def preprocess_dataset(source_path):
    lines = open(source_path,'r',encoding='utf-8').read().split('\n')
    input_vectors = []
    target_y = []
    for line in lines:
        line = line.strip()
        if line == "": continue
        array = line.split('|')
        fact = array[1]
        ft = array[2]
        y = int(array[3])
        input_vectors.append(vector_text(fact+ft))
        target_y.append(y)

    return np.array(input_vectors), np.array(target_y)

def RF(train_path, test_path, model_path):
    input_vectors, target_y = preprocess_dataset(train_path)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(input_vectors,target_y)
    with open(model_path,'wb') as fw:
         pickle.dump(rf,fw)

    #test
    test_vectors,test_y = preprocess_dataset(test_path)
    pred_y = rf.predict(test_vectors)

    print(metrics.classification_report(test_y, pred_y, digits=4))

    #D@F1
    wslist = getwslist()
    assert len(pred_y) == len(wslist), ValueError("wslist != pre_y")
    wsevaluate(pred_y,test_y,wslist)

def SVM(train_path, test_path, model_path):
    input_vectors, target_y = preprocess_dataset(train_path)
    rf = SVC(kernel='linear')
    rf.fit(input_vectors, target_y)
    with open(model_path, 'wb') as fw:
        pickle.dump(rf, fw)

    # test
    test_vectors, test_y = preprocess_dataset(test_path)
    pred_y = rf.predict(test_vectors)

    print(metrics.classification_report(test_y, pred_y, digits=4))

    # D@F1
    wslist = getwslist()
    assert len(pred_y) == len(wslist), ValueError("wslist != pre_y")
    wsevaluate(pred_y, test_y, wslist)


if __name__=="__main__":
    train_path, test_path, model_path = '../../resource/gyshz_traindata/train-init.txt','../../resource/gyshz_traindata/test-init.txt','../../result/model/gyshz_RF_NLI.pkl'
    RF(train_path, test_path, model_path)



