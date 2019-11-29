import tensorflow as tf
import models.parameter as param

class Config:
    def __init__(self):

        #param v1
        # self.max_len = 70  # 这个有待统计
        # self.fnn_output = 64
        # self.dropout_rate = 0.5

        #param v2
        self.max_len = 30  # 这个有待统计
        self.fnn_output = 64
        self.dropout_rate = 0.5

        #param v3
        # self.max_len = 30  # 这个有待统计
        # self.fnn_output = 30
        # self.dropout_rate = 0.5

        #param v4
        # self.max_len = 30  # 这个有待统计
        # self.fnn_output = 128
        # self.dropout_rate = 0.5

        #param v5
        self.max_len = 50  # 这个有待统计
        self.fnn_output = 64
        self.dropout_rate = 0.5


class QHFNNModel:
    def __init__(self):
        self.config = Config()
        self.inputX = tf.placeholder(dtype=tf.float32,shape=[None,self.config.max_len,param.BaseConfig.law_word_dimension],name="inputX")
        self.y = tf.placeholder(dtype=tf.int32,shape=[None,2],name="inputY")
        self.dropout_rate = tf.placeholder(dtype=tf.float32,name="dropout_rate")

        self.run_model()

    def run_model(self):
        with tf.variable_scope("flat-layer"):
            self.flatX = tf.reduce_mean(self.inputX,axis=-1)

        with tf.variable_scope("predict-layer"):
            self.output_ = tf.nn.relu(tf.layers.dense(inputs=self.flatX,units=self.config.fnn_output,name='fnn1'))
            self.output_ = tf.layers.dropout(self.output_,rate=self.dropout_rate)
            self.logit = tf.layers.dense(inputs=self.output_,units=2,name='fnn2')

        with tf.variable_scope("optimize-layer"):
            self.pred_y = tf.argmax(tf.nn.softmax(self.logit), 1)
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit,
                                                                    labels=self.y)  # 对logits进行softmax操作后，做交叉墒，输出的是一个向量
            self.loss = tf.reduce_mean(cross_entropy)  # 将交叉熵向量求和，即可得到交叉熵
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=param.BaseConfig.lr).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.y, 1),
                                    self.pred_y)  # 由于input_y也是onehot编码，因此，调用tf.argmax(self.input_y)得到的是1所在的下表
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))




