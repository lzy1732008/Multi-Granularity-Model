#对比实验：验证MGCQ去除fusion-layer-0
import tensorflow as tf
import models.parameter as param
from models.modules import Interaction
import numpy as np

class ModelConfig:
    # v1
    X_maxlen = 30
    Y_maxlen = 50
    dropout_rate = 0.5
    TRIGRAM_D = 1000




class DSSM:
    def __init__(self):
        self.config = ModelConfig()
        self.input_X1 = tf.placeholder(name="inputX1_word", dtype=tf.float32,
                                          shape=[None, self.config.TRIGRAM_D])
        self.input_X2 = tf.placeholder(name="inputX2_word", dtype=tf.float32,
                                          shape=[None, self.config.TRIGRAM_D])
        self.y = tf.placeholder(name="target_y", dtype=tf.int32, shape=[None, 2])
        self.x2_label = tf.placeholder(name="inputX2_label", dtype=tf.float32,
                                       shape=[None, self.config.Y_maxlen, 2])
        self.dropout_rate = tf.placeholder(tf.float32, name='keep_prob')

        self.build_model()

    def build_model(self):
        # 第一层输出维度
        L1_N = 300
        l1_par_range = np.sqrt(6.0 / (TRIGRAM_D + L1_N))
        weight1 = tf.Variable(tf.random_uniform([TRIGRAM_D, L1_N],
                                                -l1_par_range,
                                                l1_par_range))
        bias1 = tf.Variable(tf.random_uniform([L1_N],
                                              -l1_par_range,
                                              l1_par_range))

        # 因为数据比较稀疏，所以用sparse_tensor_dense_matmul
        query_l1 = tf.sparse_tensor_dense_matmul(query_batch, weight1) + bias1
        doc_l1 = tf.sparse_tensor_dense_matmul(doc_batch, weight1) + bias1

        # 激活层，也可以换成别的激活函数
        query_l1_out = tf.nn.tanh(query_l1)
        doc_l1_out = tf.nn.tanh(doc_l1)




