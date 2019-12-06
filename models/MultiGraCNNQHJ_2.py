#该模型是NLI部分模型，采用的架构是法条和事实分别输入进两层1D CNN，每层都做对齐操作,
# 其中对齐操作使用的是attention-over-attention，输出法条的每个词的权重，
# 融入知识的方式：eij = ai * bj + lambda * ksj
import tensorflow as tf
import models.parameter as param
import models.modules as modules

class MultiGraConfig:
    def __init__(self):
        # initparam
        self.X_maxlen = 30
        self.Y_maxlen = 30
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
        self.x2_label = tf.placeholder(name="inputX2_label", dtype=tf.float32,
                                          shape=[None, self.config.Y_maxlen])
        self.y = tf.placeholder(name="target_y", dtype=tf.int32, shape=[None, 2])

        self.dropout_rate = tf.placeholder(tf.float32, name='keep_prob')

        self.build_model()

    def build_model(self):

        #WIL-1  word-Interaction-layer
        # with tf.variable_scope("word-Interaction-layer"):
        #     interaction = modules.Interaction(4, self.input_X1, self.input_X2)
        #     self.inter0_output_x2 = interaction.exeInteraction()

        #WIL-2  word-Interaction-layer
        # with tf.variable_scope("word-Interaction-layer"):
        #     self.beta = tf.Variable(tf.random_normal(shape=[1], stddev=0, seed=1, dtype=tf.float32), trainable=True, name='beta')
        #     interaction = modules.Interaction(5, self.input_X1, self.input_X2, self.x2_label, self.beta)
        #     self.inter0_output_x2 = interaction.exeInteraction()
            #添加一个mean pooling
            # self.inter0_output_x2 = tf.expand_dims(tf.expand_dims(self.inter0_output_x2,axis=2),axis=3)
            # self.inter0_output_x2 = tf.nn.avg_pool(self.inter0_output_x2,ksize=[1,3,1,1],strides=[1,1,1,1],padding='VALID',name='mean-pooling')

        with tf.variable_scope("first-CNN-layer"):
            self.output_x1_1 = tf.layers.conv1d(self.input_X1,filters=self.config.filters_num,kernel_size=self.config.first_kernel_size,padding='same',name='first-cnn1')
            self.output_x2_1 = tf.layers.conv1d(self.input_X2,filters=self.config.filters_num,kernel_size=self.config.first_kernel_size,padding='same',name='first-cnn2')

        with tf.variable_scope("first-interaction"):
            self.beta1 = tf.Variable(tf.random_normal(shape=[1], stddev=0, seed=1, dtype=tf.float32), trainable=True,
                                    name='beta1')
            interaction = modules.Interaction(5, self.output_x1_1,self.output_x2_1,self.x2_label,self.beta1)
            self.inter1_output_x2 = interaction.exeInteraction()
            # self.inter1_output_x2 = self.interaction(self.output_x1_1, self.output_x2_1)

            #mean pooling
            # self.inter1_output_x2 = tf.expand_dims(tf.expand_dims(self.inter1_output_x2,axis=2),axis=3)
            # self.inter1_output_x2 = tf.nn.avg_pool(self.inter1_output_x2,ksize=[1,3,1,1],strides=[1,1,1,1],padding='VALID', name='mean-pooling')
            # self.inter1_output_x2 = tf.reshape(self.inter1_output_x2, shape=[-1, 28])

        with tf.variable_scope("second-CNN-layer"):
            self.output_x1_2 = tf.layers.conv1d(self.output_x1_1,filters=self.config.filters_num,kernel_size=self.config.second_kernel_size,padding='same',name='second-cnn1')
            self.output_x2_2 = tf.layers.conv1d(self.output_x2_1,filters=self.config.filters_num,kernel_size=self.config.second_kernel_size,padding='same',name='second-cnn2')

        with tf.variable_scope("second-interaction"):
            self.beta2 = tf.Variable(tf.random_normal(shape=[1], stddev=0, seed=1, dtype=tf.float32), trainable=True,
                                    name='beta2')
            interaction = modules.Interaction(5, self.output_x1_2, self.output_x2_2,self.x2_label,self.beta2)
            self.inter2_output_x2 = interaction.exeInteraction()
            # self.inter2_output_x2 = self.interaction(self.output_x1_2, self.output_x2_2)

            #mean pooling
            # self.inter2_output_x2 = tf.expand_dims(tf.expand_dims(self.inter2_output_x2,axis=2),axis=3)
            # self.inter2_output_x2 = tf.nn.avg_pool(self.inter2_output_x2,ksize=[1,3,1,1],strides=[1,1,1,1],padding='VALID', name='mean-pooling')
            # self.inter2_output_x2 = tf.reshape(self.inter2_output_x2,shape=[-1,28])

        with tf.variable_scope("third-CNN-layer"):
            self.output_x1_3 = tf.layers.conv1d(self.output_x1_2,filters=self.config.filters_num,kernel_size=self.config.third_kernel_size,padding='same',name='third-cnn1')
            self.output_x2_3 = tf.layers.conv1d(self.output_x2_2,filters=self.config.filters_num,kernel_size=self.config.third_kernel_size,padding='same',name='third-cnn2')

        with tf.variable_scope("third-interaction"):
            self.beta3 = tf.Variable(tf.random_normal(shape=[1], stddev=0, seed=1, dtype=tf.float32), trainable=True,
                                    name='beta3')
            interaction = modules.Interaction(5, self.output_x1_3,self.output_x2_3,self.x2_label,self.beta3)
            self.inter3_output_x2 = interaction.exeInteraction()
            # self.inter3_output_x2 = self.interaction(self.output_x1_3,self.output_x2_3)

            #mean pooling
            # self.inter3_output_x2 = tf.expand_dims(tf.expand_dims(self.inter3_output_x2,axis=2),axis=3)
            # self.inter3_output_x2 = tf.nn.avg_pool(self.inter3_output_x2,ksize=[1,3,1,1],strides=[1,1,1,1],padding='VALID', name='mean-pooling')
            # self.inter3_output_x2 = tf.reshape(self.inter3_output_x2, shape=[-1, 28])


        with tf.variable_scope("fusion-layer"):
            #v2
            # self.fusion_output = tf.stack(
            #     [self.inter1_output_x2, self.inter2_output_x2, self.inter3_output_x2],
            #     axis=-1)  # [Batch,len,3]
            # self.fusion_output = tf.layers.conv1d(self.output_x1_1, filters=64,
            #                                       kernel_size=self.config.second_kernel_size, padding='same',
            #                                       name='fifth-cnn1')
            # self.fusion_output = tf.reduce_max(self.fusion_output, axis=-1)

            #v3
            self.fusion_output = tf.stack([self.inter1_output_x2, self.inter2_output_x2,self.inter3_output_x2,self.x2_label], axis=-1)  # [Batch,len,3]
            self.fusion_output = tf.layers.conv1d(self.output_x1_1,filters=64,kernel_size=self.config.second_kernel_size,padding='same',name='fifth-cnn1')
            self.fusion_output = tf.reduce_max(self.fusion_output,axis=-1)

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

    def interaction(self,inputX1,inputX2):
        len1 = inputX1.get_shape().as_list()[1]
        len2 = inputX2.get_shape().as_list()[1]
        dot_matrix = tf.matmul(inputX1,tf.transpose(inputX2,perm=[0,2,1])) #[Batch, len1,len2]
        #首先分别做row and col softmax
        x1_2_x2 = tf.nn.softmax(dot_matrix,axis=2) #x1对x2每个词的关注度
        x2_2_x1 = tf.nn.softmax(dot_matrix,axis=1) #x2对x1每个词的关注度

        #计算x1每个词获取的总weight
        x1_weight = tf.reduce_mean(x2_2_x1,axis=2) #[Batch, len1]

        #计算x2最后获取的每个词的总的weight
        x1_weight_ = tf.expand_dims(x1_weight,axis=1) #[Batch, 1, len1]
        x2_weight = tf.matmul(x1_weight_, x1_2_x2)  #[Batch, 1, len2]
        x2_weight = tf.reshape(x2_weight,shape=[-1,len2])
        return x2_weight





