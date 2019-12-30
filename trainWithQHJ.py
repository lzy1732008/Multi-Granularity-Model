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

from models.MGC_15 import *
from preps.data_load_generic import *
import models.parameter as param
from util.feedDict import feed_data_1 as feed_data

class basicPath:
    def __init__(self,time):
        self.save_dir = 'result/model/MGC_15'  # 修改处
        self.param_des = 'v3-fixdropoutbug-addstp-fixpadding' + str(time) +'times'
        self.save_path = os.path.join(self.save_dir, self.param_des + '/checkpoints/best_validation')
        self.tensorboard_dir = os.path.join(self.save_dir, self.param_des + '/tensorboard')

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)



config = MultiGraConfig()
model = MultiGranularityCNNModel()



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# def feed_data(a_word,b_word,c_word,y_batch,dropout_rate):
#     feed_dict = {
#         model.input_X1: a_word,
#         model.input_X2: b_word,
#         model.x2_label: c_word,
#         model.y: y_batch,
#         model.dropout_rate: dropout_rate,
#     }
#
#     return feed_dict



def evaluate(sess,a_word,b_word,c_word, y):
    """评估在某一数据上的准确率和损失"""
    data_len = len(a_word)
    batch_eval = get_batch_data_test(a_word, b_word,c_word, y, batch_size=param.BaseConfig.batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for a_word_batch, b_word_batch,c_word_batch, y_batch in batch_eval:
        batch_len = len(a_word_batch)
        feed_dict = feed_data(model,a_word_batch, b_word_batch,c_word_batch, y_batch,1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train(train_data, val_data,Path):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖

    if not os.path.exists(Path.tensorboard_dir):
        os.makedirs(Path.tensorboard_dir)

    #结果可视化与存储
    tf.summary.scalar("loss", model.loss) #可视化loss
    tf.summary.scalar("accuracy", model.acc)  #可视化acc
    merged_summary = tf.summary.merge_all()   #将所有操作合并输出
    writer = tf.summary.FileWriter(Path.tensorboard_dir) #将summary data写入磁盘

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(Path.save_dir):
        os.makedirs(Path.save_dir)



    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()

    train_x1_word, train_x2_word, train_x2_label, train_y = train_data
    val_x1_word,  val_x2_word, val_x2_label, val_y = val_data

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
        batch_train = get_batch_data(train_x1_word, train_x2_word, train_x2_label, train_y, batch_size=param.BaseConfig.batch_size)
        for a_word_batch, b_word_batch, c_word_batch, y_batch in batch_train:
            feed_dict = feed_data(model,a_word_batch, b_word_batch, c_word_batch,y_batch,model.config.dropout_rate)

            if total_batch % param.BaseConfig.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % param.BaseConfig.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能

                feed_dict[model.dropout_rate] = 1.0
                loss_train, acc_train,pre_y, logit, true_y = session.run([model.loss, model.acc,model.pred_y,model.logit,model.y], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, val_x1_word,  val_x2_word,  val_x2_label, val_y)  # 验证当前会话中的模型的loss和acc


                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=Path.save_path)
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

def test(test_data, Path):
    #载入随机森林模型

    print("Loading test data...")
    start_time = time.time()

    test_x1_word,  test_x2_word, test_x2_label, test_y = test_data


    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=Path.save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, test_x1_word, test_x2_word, test_x2_label, test_y)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = param.BaseConfig.batch_size
    data_len = len(test_x1_word)
    num_batch = int((data_len) / batch_size)
    # num_batch = 1

    y_test_cls = np.argmax(test_y, 1)
    y_pred_cls = np.zeros(shape=data_len, dtype=np.int32)  # 保存预测结果
    probs = np.zeros(shape=[data_len,2],dtype=np.float32)

    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = feed_data(model,test_x1_word[start_id:end_id],test_x2_word[start_id:end_id],test_x2_label[start_id:end_id],test_y,1.0)
        # feed_dict = {
        #     model.input_X1: test_x1_word[start_id:end_id],
        #     model.input_X2: test_x2_word[start_id:end_id],
        #     model.x2_label: test_x2_label[start_id:end_id],
        #     model.y: test_y,
        #     model.dropout_rate: 1.0   #这个表示测试时不使用dropout对神经元过滤
        # }
        y_pred_cls[start_id:end_id],probs[start_id:end_id] = session.run([model.pred_y,model.logit],feed_dict=feed_dict)   #将所有批次的预测结果都存放在y_pred_cls中
        # inter_1, pool_1,pool_2,pool_3 = session.run([model.inter_1,model.fusion_output_max_1,model.fusion_output_max_2,model.fusion_output_max_3],
        #                                                             feed_dict=feed_dict)
        # print('pooling 1....')
        # print(pool_1)
        # print('pooling 2....')
        # print(pool_2)
        # print('pooling 3....')
        # print(pool_3)
        # print('inter 1.....')
        # print(inter_1)



    print(y_pred_cls)
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls,digits=4))#直接计算准确率，召回率和f值

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


    checkPrediction(y_pred_cls,y_test_cls,probs)
    # print("beta value", beta1,beta2,beta3)
    #check error prediction
    # print(y_pred_cls)
    #

    return y_test_cls,y_pred_cls

import json
def checkPrediction(pred_cls, target_y,probs):
    test_content = open('resource/test-init.txt','r',encoding='utf-8').read()
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
            assert y == target_y[index],ValueError(s+str(target_y[index]))
            if target_y[index] == pred_cls[index]: right.append(s)
            if target_y[index] == 0 and pred_cls[index] == 1: wrong.append(s)
            index += 1

    with open('resource/预测结果分析/MGC_15predictAna-使用修改后的dropout.json','w',encoding='utf-8') as fw:
        json.dump(law_result,fw)

    # print('predction is right.......')
    # print('\n'.join(right))
    # print('prediction is wrong')
    # print('\n'.join(wrong))


def run_mutli():
    # 载入随机森林模型
    with open(param.BaseConfig.rf_model_path, 'rb') as fr:
        rf = pickle.load(fr)
    train_data, val_data, test_data = data_load(param.BaseConfig.trainPath, param.BaseConfig.valPath, param.BaseConfig.testPath, model, rf)
    # train_data, val_data, test_data = data_load(None, None,
    #                                             param.BaseConfig.testPath, model, rf)
    print('train data shape:{0}\n val data shape:{1}\n test data shape:{2}'.format(len(train_data), len(val_data), len(test_data)))
    for i in range(1,2):
        Path = basicPath(i)
        train(train_data,val_data,Path)
        test(test_data, Path)


    # for j in range(3):
    #     print("the {0}nd testing......".format(str(j)))
    #     Path = basicPath(j)
    #     test(test_data, Path)



run_mutli()
# wsnamels = getwslist(model=model)
# wsevaluate(y_test_cls, y_pred_cls,wsnamels)