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

        # 第二层的输出维度
        L2_N = 300
        l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))
        weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N],
                                                -l2_par_range,
                                                l2_par_range))
        bias2 = tf.Variable(tf.random_uniform([L2_N],
                                              -l2_par_range,
                                              l2_par_range))

        query_l2 = tf.sparse_tensor_dense_matmul(query_l1_out, weight2) + bias2
        doc_l2 = tf.sparse_tensor_dense_matmul(doc_l1_out, weight2) + bias2

        query_l2_out = tf.nn.tanh(query_l2)
        doc_l2_out = tf.nn.tanh(doc_l2)

        # 第三层
        L3_N = 128
        l3_par_range = np.sqrt(6.0 / (L2_N + L3_N))
        weight3 = tf.Variable(tf.random_uniform([L2_N, L3_N],
                                                -l3_par_range,
                                                l3_par_range))
        bias3 = tf.Variable(tf.random_uniform([L3_N],
                                              -l3_par_range,
                                              l3_par_range))

        query_l3 = tf.sparse_tensor_dense_matmul(query_l2_out, weight3) + bias3
        doc_l3 = tf.sparse_tensor_dense_matmul(doc_l2_out, weight3) + bias3

        query_l3_out = tf.nn.tanh(query_l3)
        doc_l3_out = tf.nn.tanh(doc_l3)

        # NEG表示负样本的个数
        NEG = 4

        # ||yq||
        query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_l3_out), 1, True)),
                             [NEG + 1, 1])
        # ||yd||
        doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_l3_out), 1, True))

        # yqT yd
        prod = tf.reduce_sum(tf.mul(tf.tile(query_l3_out, [NEG + 1, 1]), doc_l3_out), 1, True)
        norm_prod = tf.mul(query_norm, doc_norm)

        # cosine
        cos_sim_raw = tf.truediv(prod, norm_prod)
        cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, BS])) * Gamma

        # BS为batch_size，计算batch平均损失

        prob = tf.nn.softmax((cos_sim))

        # 正例的softmax值
        hit_prob = tf.slice(prob, [0, 0], [-1, 1])

        # 最小化loss，计算batch的平均损失
        loss = -tf.reduce_sum(tf.log(hit_prob)) / BS

        # 定义优化方法和学习率
        train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

        with tf.Session(config=config) as sess:
            sess.run(tf.initialize_all_variables())
            for step in range(FLAGS.max_steps):
                sess.run(train_step, feed_dict={query_batch: ...
                                                doc_batch: ...}})




