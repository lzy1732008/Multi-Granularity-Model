# 对比试验：只使用融合层3的结果输入到预测层中
import tensorflow as tf
import models.parameter as param
from models.modules import Interaction

class MultiGraConfig:
    # v1
    X_maxlen = 30
    Y_maxlen = 50
    dropout_rate = 0.5
    first_kernel_size = 2
    second_kernel_size = 4
    third_kernel_size = 8
    filters_num = 100
    mlp_output = 64




class MultiGranularityCNNModel:
    def __init__(self):
        self.config = MultiGraConfig()
        self.input_X1 = tf.placeholder(name="inputX1_word", dtype=tf.float32,
                                          shape=[None, self.config.X_maxlen, param.BaseConfig.word_dimension])
        self.input_X2 = tf.placeholder(name="inputX2_word", dtype=tf.float32,
                                          shape=[None, self.config.Y_maxlen, param.BaseConfig.word_dimension])
        self.y = tf.placeholder(name="target_y", dtype=tf.int32, shape=[None, 2])
        self.x2_label = tf.placeholder(name="inputX2_label", dtype=tf.float32,
                                       shape=[None, self.config.Y_maxlen, 2])
        self.dropout_rate = tf.placeholder(tf.float32, name='keep_prob')

        self.build_model()

    def build_model(self):
        # with tf.variable_scope("related-score-layer"):
        #     beta = tf.Variable(tf.random_normal(shape=[param.BaseConfig.word_dimension, 2], stddev=0, seed=1, dtype=tf.float32), trainable=True,
        #                        name='beta')
        #     self.weigt = tf.einsum('abc,cd->abd', self.input_X2, beta)  # [B,l2,2]
        #     self.weigt = self.weigt * self.x2_label
            # max_num = tf.keras.backend.repeat_elements(tf.reduce_max(self.weigt,axis=-1,keep_dims=True),rep=2,axis=-1)# [B,l2,2]
            # min_num = tf.keras.backend.repeat_elements(tf.reduce_min(self.weigt,axis=-1,keep_dims=True),rep=2,axis=-1)# [B,l2,2]
            # self.weigt = (self.weigt - min_num + 0.1)/(max_num - min_num + 0.2)# [B,l2,2]
            # self.weigt = tf.reduce_sum(self.weigt * self.x2_label,axis=-1) # [B,l2]

            # ks = tf.reduce_sum(self.weigt * self.x2_label, axis=-1)  # [B,l2]
            # self.ks_rep = tf.reshape(tf.keras.backend.repeat_elements(ks, rep=self.config.X_maxlen, axis=1),
            #                      shape=[-1, self.config.X_maxlen, self.config.Y_maxlen])

            # self.weigt = tf.sigmoid(tf.einsum('abc,cd->abd', self.input_X2, beta))  # [B,l2,2]
            # self.weight = tf.reduce_sum(weigt * self.x2_label, axis=-1)  # [B,l2]
            # self.ks_rep = tf.reshape(tf.keras.backend.repeat_elements(self.weigt, rep=self.config.X_maxlen, axis=1),
            #                      shape=[-1, self.config.X_maxlen, self.config.Y_maxlen])



        with tf.variable_scope("first-CNN-layer"):
            self.output_x1_1 = tf.layers.conv1d(self.input_X1,filters=self.config.filters_num,kernel_size=self.config.first_kernel_size,padding='same',name='first-cnn1')
            self.output_x2_1 = tf.layers.conv1d(self.input_X2,filters=self.config.filters_num,kernel_size=self.config.first_kernel_size,padding='same',name='first-cnn2')



        with tf.variable_scope("second-CNN-layer"):
            self.output_x1_2 = tf.layers.conv1d(self.output_x1_1,filters=self.config.filters_num,kernel_size=self.config.second_kernel_size,padding='same',name='second-cnn1')
            self.output_x2_2 = tf.layers.conv1d(self.output_x2_1,filters=self.config.filters_num,kernel_size=self.config.second_kernel_size,padding='same',name='second-cnn2')



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

            self.inter_3 = self.interaction(self.output_x1_3,self.output_x2_3)
            self.inter_rep_3 = tf.reshape(tf.keras.backend.repeat_elements(self.inter_3, rep=param.BaseConfig.word_dimension, axis=1),shape=[-1,self.config.Y_maxlen,param.BaseConfig.word_dimension])

        with tf.variable_scope("fusion-layer-3"):
            self.x2_inter_3 = self.inter_rep_3 * self.input_X2
            self.fusion_output_3 = tf.concat([self.input_X2,self.x2_inter_3,self.input_X2 - self.x2_inter_3, self.input_X2 * self.x2_inter_3], axis=-1)  # [Batch, len, 2 + 4 * dimension]
            self.fusion_output_3 = tf.layers.dense(inputs=self.fusion_output_3,units=self.config.mlp_output,name='fusion-fnn')
            # self.fusion_output_3 = tf.layers.dropout(self.fusion_output_3, rate=self.dropout_rate)
            self.fusion_output_max_3 = tf.reduce_max(self.fusion_output_3,axis=-1) #[B,l]


        with tf.variable_scope("predict-layer"):
            self.output_1 = tf.nn.relu(tf.layers.dense(inputs=self.fusion_output_max_3,units=self.config.mlp_output,name='fnn1'))
            self.output_1 = tf.layers.dropout(self.output_1,rate=self.dropout_rate)

            self.logit = tf.layers.dense(inputs=self.output_1,units=2,name='fnn2')

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

    def interaction2(self,inputX1,inputX2):
        len1 = inputX1.get_shape().as_list()[1]
        len2 = inputX2.get_shape().as_list()[1]
        dot_matrix = tf.matmul(inputX1,tf.transpose(inputX2,perm=[0,2,1])) #[Batch, len1,len2]
        #首先分别做row and col softmax
        x1_2_x2 = tf.nn.softmax(dot_matrix,axis=2) #x1对x2每个词的关注度
        x2_2_x1 = tf.nn.softmax(dot_matrix,axis=1) #x2对x1每个词的关注度

        #计算x1每个词获取的总weight
        x1_weight = tf.reduce_sum(x2_2_x1,axis=2) #[Batch, len1]

        #计算x2最后获取的每个词的总的weight
        x1_weight_ = tf.expand_dims(x1_weight,axis=1) #[Batch, 1, len1]
        x2_weight = tf.matmul(x1_weight_, x1_2_x2)  #[Batch, 1, len2]
        x2_weight = tf.reshape(x2_weight,shape=[-1,len2])
        return x2_weight


    def interaction3(self,inputX1,inputX2):
        len1 = inputX1.get_shape().as_list()[1]
        len2 = inputX2.get_shape().as_list()[1]
        dot_matrix = tf.matmul(inputX1,tf.transpose(inputX2,perm=[0,2,1])) #[Batch, len1,len2]
        #首先分别做row and col softmax
        x1_2_x2 = tf.nn.softmax(dot_matrix,axis=2) #x1对x2每个词的关注度
        x2_2_x1 = tf.nn.softmax(dot_matrix,axis=1) #x2对x1每个词的关注度

        #计算x1每个词获取的总weight
        x1_weight = tf.reduce_max(x2_2_x1,axis=2) #[Batch, len1]

        #计算x2最后获取的每个词的总的weight
        x1_weight_ = tf.expand_dims(x1_weight,axis=1) #[Batch, 1, len1]
        x2_weight = tf.matmul(x1_weight_, x1_2_x2)  #[Batch, 1, len2]
        x2_weight = tf.reshape(x2_weight,shape=[-1,len2])
        return x2_weight




