from preps.data_load_generic import get_batch_data_test
import models.parameter as param
import numpy as np
def evaluate_1(model,sess,a_word,b_word,y,feed_data):
    """评估在某一数据上的准确率和损失"""
    data_len = len(a_word)
    batch_eval = get_batch_data_test(a_word, b_word, y, batch_size=param.BaseConfig.batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for a_word_batch, b_word_batch, y_batch in batch_eval:
        batch_len = len(a_word_batch)
        feed_dict = feed_data(model,a_word_batch, b_word_batch,y_batch,1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def evaluate_2(model,sess,a_word,b_word,seq_1, seq_2, y, feed_data):
    """评估在某一数据上的准确率和损失"""
    data_len = len(a_word)
    batch_eval = get_batch_data_test(a_word, b_word,seq_1, seq_2, y, batch_size=param.BaseConfig.batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for a_word_batch, b_word_batch,seq_1_batch, seq_2_batch, y_batch in batch_eval:
        batch_len = len(a_word_batch)
        feed_dict = feed_data(model,a_word_batch, b_word_batch,seq_1_batch, seq_2_batch, y_batch,1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def evaluate_3(model,sess,a_word,b_word,c_word, y, feed_data):
    """评估在某一数据上的准确率和损失"""
    data_len = len(a_word)
    batch_eval = get_batch_data_test(a_word, b_word,c_word, y, batch_size=param.BaseConfig.batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for a_word_batch, b_word_batch,c_batch, y_batch in batch_eval:
        batch_len = len(a_word_batch)
        feed_dict = feed_data(model,a_word_batch, b_word_batch,c_batch,y_batch,1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def evaluate_4(model,sess,a_word,b_word,c_word, d_word, y, feed_data):
    """评估在某一数据上的准确率和损失"""
    data_len = len(a_word)
    batch_eval = get_batch_data_test(a_word, b_word,c_word, d_word, y, batch_size=param.BaseConfig.batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for a_word_batch, b_word_batch,c_batch, d_word, y_batch in batch_eval:
        batch_len = len(a_word_batch)
        feed_dict = feed_data(model,a_word_batch, b_word_batch,c_batch,d_word, y_batch,1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def wsevaluate(y_pred_cls,y_test_cls,wslist):
    print('y_pred_cls.len:',len(y_pred_cls))
    print('y_test_cls.len',len(y_test_cls))
    print('wslist.len:',len(wslist))
    pred_true = {}
    positive = {}
    pred_pos = {}

    for i in range(len(y_test_cls)):
        if pred_pos.get(wslist[i].strip()) == None:
            pred_pos[wslist[i].strip()] = 0
        if pred_true.get(wslist[i].strip()) == None:
            pred_true[wslist[i].strip()] = 0
        if positive.get(wslist[i].strip()) == None:
            positive[wslist[i].strip()] = 0

        if y_pred_cls[i] == 1:
            pred_pos[wslist[i]] += 1
        if y_test_cls[i] == 1:
            positive[wslist[i]] += 1
        if y_test_cls[i] == y_pred_cls[i] and y_pred_cls[i] == 1:
            pred_true[wslist[i]] += 1

    F1_ls = []
    wslist = list(set(wslist))
    for wsname in wslist:
        # print(pred_pos[wsname.strip()],positive[wsname.strip()],pred_true[wsname.strip()])
        if positive[wsname.strip()] == 0:
            print('Failed')
            continue
        else:
            recall = pred_true[wsname.strip()] / (positive[wsname.strip()])
            if pred_pos[wsname.strip()] == 0:
               precision = 0
            else:
               precision = pred_true[wsname.strip()]/(pred_pos[wsname.strip()])
            if recall + precision == 0:
                F1 = 0
            else:
                F1 = (2*recall*precision)/(precision+recall)
            F1_ls.append(F1)
            # print('F1:',F1)
    print('Document F1:',np.mean(np.array(F1_ls)))


