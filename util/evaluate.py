from preps.data_load_generic import get_batch_data_test
import models.parameter as param

def evaluate1(model,sess,a_word,b_word,y,feed_data):
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


def evaluate2(model,sess,a_word,b_word,seq_1, seq_2, y, feed_data):
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

def evaluate3(model,sess,a_word,b_word,c_word, y, feed_data):
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


