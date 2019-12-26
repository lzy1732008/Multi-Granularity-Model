# 该模型是NLI部分模型，采用的架构是法条和事实分别输入进两层1D CNN，每层都做对齐操作
import tensorflow as tf
import models.parameter as param
from models.modules import Interaction
import math


class MultiGraConfig:
    # v1
    X_maxlen = 30
    Y_maxlen = 50
    dropout_rate = 0.5
    first_kernel_size = 2
    second_kernel_size = 4
    third_kernel_size = 8
    filters_num = param.BaseConfig.word_dimension
    mlp_output = 64



class MultiGranularityCNNModel:
    def __init__(self):
        self.config = MultiGraConfig()
        self.input_X1 = tf.placeholder(name="inputX1_word", dtype=tf.float32,
                                       shape=[None, self.config.X_maxlen, param.BaseConfig.word_dimension])
        self.input_X2 = tf.placeholder(name="inputX2_word", dtype=tf.float32,
                                       shape=[None, self.config.Y_maxlen, param.BaseConfig.word_dimension])
        self.align_matrix = tf.placeholder(name="align_maltrx",shape=[None, self.config.X_maxlen, self.config.Y_maxlen],
                                       dtype=tf.float32)
        self.y = tf.placeholder(name="target_y", dtype=tf.int32, shape=[None, 2])

        self.dropout_rate = tf.placeholder(tf.float32, name='keep_prob')

        self.build_model()

    def build_model(self):
        with tf.variable_scope("first-CNN-layer"):
            self.output_x1_1 = tf.layers.conv1d(self.input_X1, filters=self.config.filters_num,
                                                kernel_size=self.config.first_kernel_size, padding='same',
                                                name='first-cnn1')
            self.output_x2_1 = tf.layers.conv1d(self.input_X2, filters=self.config.filters_num,
                                                kernel_size=self.config.first_kernel_size, padding='same',
                                                name='first-cnn2')

        with tf.variable_scope("first-interaction"):
            # self.inter_1 = self.interactionSuper(self.output_x1_1, self.output_x2_1, self.q1_mask, self.q2_mask)
            self.inter_1 = self.interaction2(self.input_X1, self.input_X2,self.align_matrix)
            self.inter_rep_1 = tf.reshape(
                tf.keras.backend.repeat_elements(self.inter_1, rep=param.BaseConfig.word_dimension, axis=1),
                shape=[-1, self.config.Y_maxlen, param.BaseConfig.word_dimension])

        with tf.variable_scope("fusion-layer-1"):
            self.x2_inter_1 = self.inter_rep_1 * self.input_X2
            self.fusion_output_1 = tf.concat(
                [self.input_X2, self.x2_inter_1, self.input_X2 - self.x2_inter_1, self.input_X2 * self.x2_inter_1],
                axis=-1)  # [Batch, len, 4 * dimension]
            self.fusion_output_1 = tf.layers.dense(inputs=self.fusion_output_1, units=self.config.mlp_output,
                                                   name='fusion-fnn')
            self.fusion_output_max_1 = tf.reduce_max(self.fusion_output_1, axis=-1)

        with tf.variable_scope("second-CNN-layer"):
            self.output_x1_2 = tf.layers.conv1d(self.output_x1_1, filters=self.config.filters_num,
                                                kernel_size=self.config.second_kernel_size, padding='same',
                                                name='second-cnn1')
            self.output_x2_2 = tf.layers.conv1d(self.output_x2_1, filters=self.config.filters_num,
                                                kernel_size=self.config.second_kernel_size, padding='same',
                                                name='second-cnn2')

        with tf.variable_scope("second-interaction"):
            self.inter_2 = self.interaction(self.output_x1_2, self.output_x2_2)
            self.inter_rep_2 = tf.reshape(
                tf.keras.backend.repeat_elements(self.inter_2, rep=param.BaseConfig.word_dimension, axis=1),
                shape=[-1, self.config.Y_maxlen, param.BaseConfig.word_dimension])

        with tf.variable_scope("fusion-layer-2"):
            self.x2_inter_2 = self.inter_rep_2 * self.input_X2
            self.fusion_output_2 = tf.concat(
                [self.input_X2, self.x2_inter_2, self.input_X2 - self.x2_inter_2, self.input_X2 * self.x2_inter_2],
                axis=-1)  # [Batch, len, 4 * dimension]
            self.fusion_output_2 = tf.layers.dense(inputs=self.fusion_output_2, units=self.config.mlp_output,
                                                   name='fusion-fnn')
            # self.fusion_output_2 = tf.nn.top_k(input=self.fusion_output_2,k=5,sorted=False)
            # self.fusion_output_2 = tf.layers.dense(inputs=tf.concat([self.fusion_output_2[0],self.x2_label],axis=-1),units=self.config.mlp_output,name='fusion-fnn-2')
            self.fusion_output_max_2 = tf.reduce_max(self.fusion_output_2, axis=-1)

        with tf.variable_scope("third-CNN-layer"):
            self.output_x1_3 = tf.layers.conv1d(self.output_x1_2, filters=self.config.filters_num,
                                                kernel_size=self.config.third_kernel_size, padding='same',
                                                name='second-cnn1')
            self.output_x2_3 = tf.layers.conv1d(self.output_x2_2, filters=self.config.filters_num,
                                                kernel_size=self.config.third_kernel_size, padding='same',
                                                name='second-cnn2')
        with tf.variable_scope("third-interaction"):
            # interaction = Interaction(8, self.output_x1_3, self.output_x2_3, self.x2_label)
            # self.inter_3 = interaction.exeInteraction()

            self.inter_3 = self.interaction(self.output_x1_3, self.output_x2_3)
            self.inter_rep_3 = tf.reshape(
                tf.keras.backend.repeat_elements(self.inter_3, rep=param.BaseConfig.word_dimension, axis=1),
                shape=[-1, self.config.Y_maxlen, param.BaseConfig.word_dimension])

        with tf.variable_scope("fusion-layer-3"):
            self.x2_inter_3 = self.inter_rep_3 * self.input_X2
            self.fusion_output_3 = tf.concat(
                [self.input_X2, self.x2_inter_3, self.input_X2 - self.x2_inter_3, self.input_X2 * self.x2_inter_3],
                axis=-1)  # [Batch, len, 2 + 4 * dimension]
            self.fusion_output_3 = tf.layers.dense(inputs=self.fusion_output_3, units=self.config.mlp_output,
                                                   name='fusion-fnn')
            self.fusion_output_max_3 = tf.reduce_max(self.fusion_output_3, axis=-1)  # [B,l]

        with tf.variable_scope("Augment-layer"):
            self.fusion_output = tf.concat(
                [self.fusion_output_max_1, self.fusion_output_max_2, self.fusion_output_max_3],
                axis=-1)  # [B,2l]

        # with tf.variable_scope("Augment-layer"):
        #     self.inter_3 = tf.concat([self.inter_2,self.inter_3],axis=-1)

        with tf.variable_scope("predict-layer"):
            self.output_1 = tf.nn.relu(
                tf.layers.dense(inputs=self.fusion_output, units=self.config.mlp_output, name='fnn1'))
            self.output_1 = tf.layers.dropout(self.output_1, rate=self.config.dropout_rate)

            self.logit = tf.layers.dense(inputs=self.output_1, units=2, name='fnn2')

        with tf.variable_scope("optimize-layer"):
            self.pred_y = tf.argmax(tf.nn.softmax(self.logit), 1)

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

    def interaction(self, inputX1, inputX2):
        len1 = inputX1.get_shape().as_list()[1]
        len2 = inputX2.get_shape().as_list()[1]
        dot_matrix = tf.matmul(inputX1, tf.transpose(inputX2, perm=[0, 2, 1]))  # [Batch, len1,len2]
        # 首先分别做row and col softmax
        x1_2_x2 = tf.nn.softmax(dot_matrix, axis=2)  # x1对x2每个词的关注度
        x2_2_x1 = tf.nn.softmax(dot_matrix, axis=1)  # x2对x1每个词的关注度

        # 计算x1每个词获取的总weight
        x1_weight = tf.reduce_mean(x2_2_x1, axis=2)  # [Batch, len1]

        # 计算x2最后获取的每个词的总的weight
        x1_weight_ = tf.expand_dims(x1_weight, axis=1)  # [Batch, 1, len1]
        x2_weight = tf.matmul(x1_weight_, x1_2_x2)  # [Batch, 1, len2]
        x2_weight = tf.reshape(x2_weight, shape=[-1, len2])
        return x2_weight

    def interaction2(self,inputX1,inputX2, alignMatrix):
        batch1, len1, _ = inputX1.get_shape().as_list()
        batch2, len2, _ = inputX2.get_shape().as_list()

        beta = tf.Variable(tf.random_normal(shape=[1], stddev=0, seed=1, dtype=tf.float32), trainable=True,
                           name='beta')
        dot_matrix = tf.matmul(inputX1, tf.transpose(inputX2, perm=[0, 2, 1])) + beta[0] * alignMatrix# [Batch, len1,len2]
        # 首先分别做row and col softmax
        x1_2_x2 = tf.nn.softmax(dot_matrix, axis=2)  # x1对x2每个词的关注度
        x2_2_x1 = tf.nn.softmax(dot_matrix, axis=1)  # x2对x1每个词的关注度

        # 计算x1每个词获取的总weight
        x1_weight = tf.reduce_mean(x2_2_x1, axis=2)  # [Batch, len1]

        # 计算x2最后获取的每个词的总的weight
        x1_weight_ = tf.expand_dims(x1_weight, axis=1)  # [Batch, 1, len1]
        x2_weight = tf.matmul(x1_weight_, x1_2_x2)  # [Batch, 1, len2]
        x2_weight = tf.reshape(x2_weight, shape=[-1, len2])
        return x2_weight




