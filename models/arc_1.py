#该模型是NLI部分模型，采用的架构是法条和事实分别输入进两层1D CNN，每层都做对齐操作
import tensorflow as tf
import models.parameter as param
import models.modules as ms

class MultiGraConfig:
    # #initparam
    X_maxlen = 30
    Y_maxlen = 30
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
        self.input_X3 = tf.placeholder(name="inputX3_word",dtype=tf.float32,
                                          shape=[None,self.config.Y_maxlen,param.BaseConfig.word_dimension])
        self.y = tf.placeholder(name="target_y", dtype=tf.int32, shape=[None, 2])
        self.dropout_rate = tf.placeholder(tf.float32, name='keep_prob')

        self.build_model()

    def build_model(self):
        with tf.variable_scope("first-CNN-layer"):
            self.output_x1_1 = tf.layers.conv1d(self.input_X1,filters=self.config.filters_num,kernel_size=self.config.first_kernel_size,padding='same',name='first-cnn1')
            self.output_x2_1 = tf.layers.conv1d(self.input_X2,filters=self.config.filters_num,kernel_size=self.config.first_kernel_size,padding='same',name='first-cnn2')
            self.output_x3_1 = tf.layers.conv1d(self.input_X3, filters=self.config.filters_num,
                                                kernel_size=self.config.first_kernel_size, padding='same',
                                                name='first-cnn3')

        with tf.variable_scope("second-CNN-layer"):
            self.output_x1_2 = tf.layers.conv1d(self.output_x1_1,filters=self.config.filters_num,kernel_size=self.config.second_kernel_size,padding='same',name='second-cnn1')
            self.output_x2_2 = tf.layers.conv1d(self.output_x2_1,filters=self.config.filters_num,kernel_size=self.config.second_kernel_size,padding='same',name='second-cnn2')
            self.output_x3_2 = tf.layers.conv1d(self.output_x3_1, filters=self.config.filters_num,
                                                kernel_size=self.config.second_kernel_size, padding='same',
                                                name='second-cnn3')

        with tf.variable_scope("third-CNN-layer"):
            self.output_x1_3 = tf.layers.conv1d(self.output_x1_2,filters=self.config.filters_num,kernel_size=self.config.third_kernel_size,padding='same',name='third-cnn1')
            self.output_x2_3 = tf.layers.conv1d(self.output_x2_2,filters=self.config.filters_num,kernel_size=self.config.third_kernel_size,padding='same',name='third-cnn2')
            self.output_x3_3 = tf.layers.conv1d(self.output_x3_2, filters=self.config.filters_num,
                                                kernel_size=self.config.third_kernel_size, padding='same',
                                                name='third-cnn3')

            self.x1_maxpooling = tf.reduce_max(self.output_x1_3, axis=-1)
            self.x2_maxpooling = tf.reduce_max(self.output_x2_3, axis=-1)
            self.x3_maxpooling = tf.reduce_max(self.output_x3_3, axis=-1)

        with tf.variable_scope("fusion-layer"):
            self.fusion_output_12 = tf.concat([self.x1_maxpooling, self.x2_maxpooling],axis=-1)
            self.fusion_output_13 = tf.concat([self.x1_maxpooling, self.x3_maxpooling], axis=-1)

        with tf.variable_scope("predict-layer"):
            self.output_12 = tf.nn.relu(tf.layers.dense(inputs=self.fusion_output_12,units=self.config.mlp_output,name='fnn1'))
            self.output_12 = tf.layers.dropout(self.output_12,rate=self.config.dropout_rate)

            self.output_13 = tf.nn.relu(
                tf.layers.dense(inputs=self.fusion_output_13, units=self.config.mlp_output, name='fnn2'))
            self.output_13 = tf.layers.dropout(self.output_13, rate=self.config.dropout_rate)

            self.logit = tf.layers.dense(inputs=tf.concat([self.output_12,self.output_13],axis=-1),units=2,name='fnn3')

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





