# coding: utf-8
#包含5个输入关键参数：inputx,inputx_len,inputy,inputy_len,y
from __future__ import print_function

import time
from datetime import timedelta
import tensorflow as tf
from sklearn import metrics
import os
import sys
import pickle
import numpy as np

from models.MGCQ_23 import MultiGranularityCNNModel,MultiGraConfig   #修改处
from preps.data_load import data_load_lawone,data_load_test_lawone
from preps.data_load_generic import get_batch_data,get_batch_data_test
from util.feedDict import feed_data_4 as feed_data_fun  #修改处
from util.evaluate import evaluate_1 as evaluate_fun

import models.parameter as param

save_dir = 'result/model/MGCQ_23'  #修改处
param_des = 'v1'
# param_des = 'initparam-qj'
save_path = os.path.join(save_dir,param_des+'/checkpoints/best_validation')
tensorboard_dir = os.path.join(save_dir,param_des+'/tensorboard')


model = MultiGranularityCNNModel()

with open(param.BaseConfig.rf_model_path, 'rb') as fr:
    rf = pickle.load(fr)

qhj_label = 0

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    #结果可视化与存储
    tf.summary.scalar("loss", model.loss) #可视化loss
    tf.summary.scalar("accuracy", model.acc)  #可视化acc
    merged_summary = tf.summary.merge_all()   #将所有操作合并输出
    writer = tf.summary.FileWriter(tensorboard_dir) #将summary data写入磁盘

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    train_data, test_data, val_data = data_load_lawone(param.BaseConfig.trainPath,param.BaseConfig.valPath,param.BaseConfig.testPath,model,rfModel=rf,flag=qhj_label)
    # train_data, test_data, val_data = data_load(param.BaseConfig.trainPath, param.BaseConfig.valPath, None,
    #                                                    model)

    train_x1_word, train_x2_word,  train_y = train_data
    val_x1_word,  val_x2_word,  val_y = val_data

    # train_x1_word, train_x2_word, train_align, train_y = train_data
    # val_x1_word,  val_x2_word, val_align, val_y = val_data

    # train_x1_word, train_x2_word,  train_seq_1, train_seq_2, train_y = train_data
    # val_x1_word,  val_x2_word, val_seq_1, val_seq_2, val_y = val_data



    print('train len',len(train_x1_word))
    print('val_len',len(val_x1_word))

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(param.BaseConfig.num_epochs):
        print('Epoch:', epoch + 1)
        # batch_train = get_batch_data(train_x1_word, train_x2_word, train_align, train_y,
        #                              batch_size=param.BaseConfig.batch_size)
        batch_train = get_batch_data(train_x1_word, train_x2_word, train_y,
                                     batch_size=param.BaseConfig.batch_size)
        # batch_train = get_batch_data(train_x1_word, train_x2_word,  train_seq_1, train_seq_2, train_y, batch_size=param.BaseConfig.batch_size)


        # for a_word_batch, b_word_batch, seq_1_batch, seq_2_batch, y_batch in batch_train:
        for a_word_batch, b_word_batch, y_batch in batch_train:
        # for a_word_batch, b_word_batch, c_batch, y_batch in batch_train:
            # feed_dict = feed_data(model,a_word_batch,b_word_batch,seq_1_batch,seq_2_batch,y_batch,model.config.dropout_rate)
            feed_dict = feed_data_fun(model, a_word_batch, b_word_batch, y_batch,model.config.dropout_rate)
            # feed_dict = feed_data_fun(model, a_word_batch, b_word_batch, c_batch, y_batch, model.config.dropout_rate)
            if total_batch % param.BaseConfig.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % param.BaseConfig.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能

                feed_dict[model.dropout_rate] = 1.0
                loss_train, acc_train,pre_y, logit, true_y, = session.run([model.loss, model.acc,model.pred_y,model.logit,model.y], feed_dict=feed_dict)
                # loss_val, acc_val = evaluate(session, val_x1_word,  val_x2_word, val_seq_1, val_seq_2, val_y )  # 验证当前会话中的模型的loss和acc
                # loss_val, acc_val = evaluate_fun(model,session, val_x1_word, val_x2_word, val_align, val_y,feed_data_fun)
                loss_val, acc_val = evaluate_fun(model, session, val_x1_word, val_x2_word, val_y,feed_data_fun)


                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0}, Train Loss: {1}, Train Acc: {2},' \
                      + ' Val Loss: {3}, Val Acc: {4}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break

def test():
    print("Loading test data...")
    start_time = time.time()
    test_data = data_load_test_lawone(model,rf,flag=qhj_label)
    # test_data = data_load_test(model)
    test_x1_word,  test_x2_word,  test_y = test_data
    # test_x1_word, test_x2_word, test_seq_1, test_seq_2, test_y = test_data
    # test_x1_word, test_x2_word, test_align, test_y = test_data


    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    # loss_test, acc_test = evaluate(session, test_x1_word, test_x2_word, test_seq_1, test_seq_2, test_y)
    # loss_test, acc_test = evaluate_fun(model,session, test_x1_word, test_x2_word, test_y, feed_data_fun)
    # loss_test, acc_test = evaluate_fun(model, session, test_x1_word, test_x2_word, test_align, test_y, feed_data_fun)
    loss_test, acc_test = evaluate_fun(model, session, test_x1_word, test_x2_word,test_y, feed_data_fun)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = param.BaseConfig.batch_size
    data_len = len(test_x1_word)
    num_batch = int((data_len) / batch_size)
    # num_batch = 1

    y_test_cls = np.argmax(test_y, 1)
    y_pred_cls = np.zeros(shape=data_len, dtype=np.int32)  # 保存预测结果
    probs = np.zeros(shape=[data_len,2], dtype=np.float32)
    inter_1,inter_2,inter_3 = [],[],[]
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        # feed_dict = feed_data(model, test_x1_word[start_id:end_id], test_x2_word[start_id:end_id],
        #                       test_seq_1[start_id:end_id], test_seq_2[start_id:end_id],test_y, 1.0)
        # feed_dict = feed_data_fun(model, test_x1_word[start_id:end_id], test_x2_word[start_id:end_id],
        #                    test_align[start_id:end_id],test_y, 1.0)

        feed_dict = feed_data_fun(model, test_x1_word[start_id:end_id], test_x2_word[start_id:end_id],test_y, 1.0)

        y_pred_cls[start_id:end_id],probs[start_id:end_id] = session.run([model.pred_y,model.logit], feed_dict=feed_dict)   #将所有批次的预测结果都存放在y_pred_cls中
        # inter_3,logit = session.run([model.fusion_output_max_3,model.logit],
        #                                                                 feed_dict=feed_dict)
        # print('inter.......')
        # print(inter_3)
        # print(logit)



    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls,digits=4))#直接计算准确率，召回率和f值

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    # print('prediction...')
    # print(y_pred_cls)

    # time_dif = get_time_dif(start_time)
    # print("Time usage:", time_dif)
    #
    # print('predict..')
    # print(y_pred_cls)
    #
    # print('inetr1...')
    # print(inter_1)
    #
    # print('inter2...')
    # print(inter_2)
    #
    # print('inter3....')
    # print(inter_3)
    # checkPrediction(y_pred_cls,y_test_cls,probs)
    return y_test_cls,y_pred_cls
import json
def checkPrediction(pred_cls, target_y,probs):
    test_content = open('resource/test-qj.txt','r',encoding='utf-8').read()
    lines = test_content.split('\n')
    index = 0
    right = []
    wrong = []
    result = []
    law_result = {}
    for line in lines:
        line = line.strip()
        if line != '':
            items = line.split('|')
            assert len(items) == 4, ValueError("The number of items in this line is less than 4, content:" + line)
            fact = items[1]
            law = items[2]
            y = int(items[-1])
            s = 'fact:{0}, law:{1}, pred:{2}, y:{3}'.format(fact,law,pred_cls[index], y)
            # result.append([fact,law,pred_cls[index],y,probs[index]])
            if law not in law_result.keys():
                law_result[law] = {}
            law_result[law][fact] = [int(pred_cls[index]),int(y),list(map(str,list(probs[index])))]
            assert y == target_y[index],ValueError(s)
            # if target_y[index] == pred_cls[index]: right.append(s)
            # else:wrong.append(s)
            index += 1

    with open('resource/预测结果分析/MGCQ_23-qj.json','w',encoding='utf-8') as fw:
        json.dump(law_result,fw)

train()
y_test_cls,y_pred_cls = test()

# data_load_lawone(param.BaseConfig.trainPath,param.BaseConfig.valPath,param.BaseConfig.testPath,model,rfModel=rf,flag=qhj_label)
# data_load_lawone(param.BaseConfig.trainPath,param.BaseConfig.valPath,param.BaseConfig.testPath,model,rfModel=rf,flag=qhj_label)

# wsnamels = getwslist(model=model)
# wsevaluate(y_test_cls, y_pred_cls,wsnamels)