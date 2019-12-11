
import tensorflow as tf
import models.parameter as param
import models.modules as modules

class MultiGraConfig:
    def __init__(self):
        # v1
        self.X_maxlen = 30
        self.Y_maxlen = 50
        self.dropout_rate = 0.5
        self.first_kernel_size = 2
        self.second_kernel_size = 4
        self.third_kernel_size = 8
        self.filters_num = param.BaseConfig.word_dimension
        self.mlp_output = 2 * self.Y_maxlen
        self.knowledge_dimension = param.BaseConfig.word_dimension

class MultiGranularityCNNModel:
    def __init__(self,config):
        self.config = config
        self.input_X1 = tf.placeholder(name="inputX1_word", dtype=tf.float32,
                                          shape=[None, self.config.X_maxlen, param.BaseConfig.word_dimension])
        self.input_X2 = tf.placeholder(name="inputX2_word", dtype=tf.float32,
                                          shape=[None, self.config.Y_maxlen, param.BaseConfig.word_dimension])
        self.y = tf.placeholder(name="target_y", dtype=tf.int32, shape=[None, 2])

        self.dropout_rate = tf.placeholder(tf.float32, name='keep_prob')

        self.build_model()

    def build_model(self):
        with tf.variable_scope("first-CNN-layer"):
            self.output_x1_1 = tf.layers.conv1d(self.input_X1,filters=self.config.filters_num,kernel_size=self.config.first_kernel_size,padding='same',name='first-cnn1')
            self.output_x2_1 = tf.layers.conv1d(self.input_X2,filters=self.config.filters_num,kernel_size=self.config.first_kernel_size,padding='same',name='first-cnn2')
            self.pooling_x2_1 = tf.reduce_max(self.output_x2_1,axis=-1) #[B,l]

        with tf.variable_scope("second-CNN-layer"):
            self.output_x1_2 = tf.layers.conv1d(self.output_x1_1,filters=self.config.filters_num,kernel_size=self.config.second_kernel_size,padding='same',name='second-cnn1')
            self.output_x2_2 = tf.layers.conv1d(self.output_x2_1,filters=self.config.filters_num,kernel_size=self.config.second_kernel_size,padding='same',name='second-cnn2')

        with tf.variable_scope("third-CNN-layer"):
            self.output_x1_3 = tf.layers.conv1d(self.output_x1_2,filters=self.config.filters_num,kernel_size=self.config.third_kernel_size,padding='same',name='third-cnn1')
            self.output_x2_3 = tf.layers.conv1d(self.output_x2_2,filters=self.config.filters_num,kernel_size=self.config.third_kernel_size,padding='same',name='third-cnn2')
            self.pooling_x2_3 = tf.reduce_max(self.output_x2_3,axis=-1)

        with tf.variable_scope("interaction"):
            interaction11 = modules.Interaction(4, self.output_x1_1,self.output_x2_1)
            self.inter11 = interaction11.exeInteraction()

            # interaction21 = modules.Interaction(4, self.output_x1_2, self.output_x2_1)
            # self.inter21 = interaction21.exeInteraction()

            # interaction31 = modules.Interaction(4, self.output_x1_3, self.output_x2_1)
            # self.inter31 = interaction31.exeInteraction()

            # interaction12 = modules.Interaction(4, self.output_x1_1,self.output_x2_2)
            # self.inter12 = interaction12.exeInteraction()

            interaction22 = modules.Interaction(4, self.output_x1_2, self.output_x2_2)
            self.inter22 = interaction22.exeInteraction()

            # interaction32 = modules.Interaction(4, self.output_x1_3, self.output_x2_2)
            # self.inter32 = interaction32.exeInteraction()
            #
            # interaction13 = modules.Interaction(4, self.output_x1_1,self.output_x2_3)
            # self.inter13 = interaction13.exeInteraction()
            #
            # interaction23 = modules.Interaction(4, self.output_x1_2, self.output_x2_3)
            # self.inter23 = interaction23.exeInteraction()

            interaction33 = modules.Interaction(4, self.output_x1_3, self.output_x2_3)
            self.inter33 = interaction33.exeInteraction() #[B,l]

            # self.inter = tf.reshape(tf.stack([self.inter11,self.inter12,self.inter13,self.inter12,self.inter22,self.inter32,self.inter31,self.inter32,self.inter33],axis=-1),
            #            shape=[-1,self.config.Y_maxlen,9]) #[B,9,l]
            self.inter = tf.reshape(tf.stack([self.inter11,self.inter22,self.inter33],axis=-1),shape=[-1,self.config.Y_maxlen,3])

        with tf.variable_scope("fusion-layer"):
            self.fusion_output = tf.reduce_max(self.inter,axis=1)

        with tf.variable_scope("predict-layer"):
            self.output_ = tf.nn.relu(tf.layers.dense(inputs=self.fusion_output,units=self.config.mlp_output,name='fnn1'))
            self.output_ = tf.layers.dropout(self.output_,rate=self.config.dropout_rate)
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

