#该模型是NLI部分模型，采用的架构是法条和事实分别输入进两层1D CNN，每层都做对齐操作
import tensorflow as tf
import models.parameter as param

class MultiGraConfig:
    # #initparam
    X_maxlen = 30
    Y_maxlen = 30
    dropout_rate = 0.5
    first_kernel_size = 2
    second_kernel_size = 4
    third_kernel_size = 8
    filters_num = param.BaseConfig.word_dimension
    mlp_output = 2 * Y_maxlen  #v1
    # mlp_output= 128 #v2
    # mlp_output = 64 #v3

#v2
    # X_maxlen = 30
    # Y_maxlen = 30
    # dropout_rate = 0.5
    # first_kernel_size = 2
    # second_kernel_size = 4
    # third_kernel_size = 6
    # filters_num = param.BaseConfig.word_dimension
    # mlp_output = 2 * Y_maxlen

#v3
    # X_maxlen = 30
    # Y_maxlen = 30
    # dropout_rate = 0.8
    # first_kernel_size = 2
    # second_kernel_size = 4
    # third_kernel_size = 8
    # filters_num = param.BaseConfig.word_dimension
    # mlp_output = 2 * Y_maxlen


class MultiGranularityCNNModel:
    def __init__(self):
        self.config = MultiGraConfig()
        self.input_X1 = tf.placeholder(name="inputX1_word", dtype=tf.float32,
                                          shape=[None, self.config.X_maxlen, param.BaseConfig.word_dimension])
        self.input_X2 = tf.placeholder(name="inputX2_word", dtype=tf.float32,
                                          shape=[None, self.config.X_maxlen, param.BaseConfig.word_dimension])
        self.y = tf.placeholder(name="target_y", dtype=tf.int32, shape=[None, 2])
        self.dropout_rate = tf.placeholder(tf.float32, name='keep_prob')

        self.build_model()

    def build_model(self):
        with tf.variable_scope("first-CNN-layer"):
            self.output_x1_1 = tf.layers.conv1d(self.input_X1,filters=self.config.filters_num,kernel_size=self.config.first_kernel_size,padding='same',name='first-cnn1')
            self.output_x2_1 = tf.layers.conv1d(self.input_X2,filters=self.config.filters_num,kernel_size=self.config.first_kernel_size,padding='same',name='first-cnn2')
            self.output_x21_1 = tf.layers.conv1d(self.input_X1, filters=self.config.filters_num,
                                                kernel_size=3, padding='same',
                                                name='first-cnn3')
            self.output_x21_1 = tf.reduce_max(self.output_x21_1,axis=-1)


        with tf.variable_scope("first-interaction"):
            self.inter1_output_x2 = self.interaction(self.output_x1_1,self.output_x2_1)

        with tf.variable_scope("second-CNN-layer"):
            self.output_x1_2 = tf.layers.conv1d(self.output_x1_1,filters=self.config.filters_num,kernel_size=self.config.second_kernel_size,padding='same',name='second-cnn1')
            self.output_x2_2 = tf.layers.conv1d(self.output_x2_1,filters=self.config.filters_num,kernel_size=self.config.second_kernel_size,padding='same',name='second-cnn2')


        with tf.variable_scope("second-interaction"):
            self.inter2_output_x2 = self.interaction(self.output_x1_2, self.output_x2_2)

        with tf.variable_scope("third-CNN-layer"):
            self.output_x1_3 = tf.layers.conv1d(self.output_x1_2,filters=self.config.filters_num,kernel_size=self.config.third_kernel_size,padding='same',name='third-cnn1')
            self.output_x2_3 = tf.layers.conv1d(self.output_x2_2,filters=self.config.filters_num,kernel_size=self.config.third_kernel_size,padding='same',name='third-cnn2')

            self.output_x1_3 = tf.layers.dropout(self.output_x1_3, rate=self.config.dropout_rate)
            self.output_x2_3 = tf.layers.dropout(self.output_x2_3, rate=self.config.dropout_rate)

        with tf.variable_scope("third-interaction"):
            self.inter3_output_x2 = self.interaction(self.output_x1_3,self.output_x2_3)

        with tf.variable_scope("fusion-layer"):
            self.fusion_output = tf.concat([self.inter1_output_x2, self.inter2_output_x2,self.inter3_output_x2], axis=-1)  # [Batch, 3 * len]

        with tf.variable_scope("predict-layer"):
            self.output_1 = tf.nn.relu(tf.layers.dense(inputs=self.fusion_output,units=self.config.mlp_output,name='fnn1'))
            self.output_1 = tf.layers.dropout(self.output_1,rate=self.config.dropout_rate)

            self.output_ = tf.nn.relu(tf.layers.dense(inputs=tf.concat([self.output_x21_1,self.output_1],axis=-1),units=self.config.mlp_output,name='fnn2'))
            self.output_ = tf.layers.dropout(self.output_, rate=self.config.dropout_rate)

            self.logit = tf.layers.dense(inputs=self.output_,units=2,name='fnn3')

        with tf.variable_scope("optimize-layer"):
            self.pred_y = tf.argmax(tf.nn.softmax(self.logit), 1)
            # 损失函数，交叉熵
            # 采用代价敏感损失函数
            count_class = tf.reduce_sum(self.y,axis=0)
            count_neg = tf.cast(count_class[0],dtype=tf.float32)
            count_pos = tf.cast(count_class[1],dtype=tf.float32)
            pos_weight = count_neg/count_pos
            self.target = tf.cast(self.y,dtype=tf.float32)
            cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=self.target, logits=self.logit,
                                                                     pos_weight=pos_weight)

            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit,
            #                                                         labels=self.y)  # 对logits进行softmax操作后，做交叉墒，输出的是一个向量

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




