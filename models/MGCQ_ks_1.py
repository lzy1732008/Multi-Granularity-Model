#该模型是NLI部分模型，采用的架构是法条和事实分别输入进两层1D CNN，每层都做对齐操作
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
    filters_num = param.BaseConfig.word_dimension
    mlp_output = 64



class MultiGranularityCNNModel:
    def __init__(self):
        self.config = MultiGraConfig()
        self.input_X1 = tf.placeholder(name="inputX1_word", dtype=tf.float32,
                                          shape=[None, self.config.X_maxlen, param.BaseConfig.word_dimension])
        self.input_X2 = tf.placeholder(name="inputX2_word", dtype=tf.float32,
                                          shape=[None, self.config.Y_maxlen, param.BaseConfig.word_dimension])
        self.ks = tf.placeholder(name="inputX2_ks",dtype=tf.float32,
                                          shape=[None, 3, param.BaseConfig.word_dimension])
        self.y = tf.placeholder(name="target_y", dtype=tf.int32, shape=[None, 2])
        self.x2_label = tf.placeholder(name="inputX2_label", dtype=tf.float32,
                                       shape=[None, self.config.Y_maxlen, 2])
        self.dropout_rate = tf.placeholder(tf.float32, name='keep_prob')

        self.build_model()

    def build_model(self):
        with tf.variable_scope("first-CNN-layer"):
            self.output_x1_1 = tf.layers.conv1d(self.input_X1,filters=self.config.filters_num,kernel_size=self.config.first_kernel_size,padding='same',name='first-cnn1')
            self.output_x2_1 = tf.layers.conv1d(self.input_X2,filters=self.config.filters_num,kernel_size=self.config.first_kernel_size,padding='same',name='first-cnn2')

        with tf.variable_scope("knowledge-gate"):
            self.new_x1 = self.gate1(ks=self.ks,inputx=self.output_x1_1)

        with tf.variable_scope("inputx-CNN-layer"):
            self.new_x1_ = tf.layers.conv1d(self.new_x1, filters=self.config.filters_num,
                                                kernel_size=self.config.first_kernel_size, padding='same',
                                                name='first-cnn1')
            self.new_x1_ = tf.reduce_max(self.new_x1_,axis=-1)

        with tf.variable_scope("first-interaction"):
            self.inter_1 = self.interaction(self.output_x1_1, self.output_x2_1)
            self.inter_rep_1 = tf.reshape(
                tf.keras.backend.repeat_elements(self.inter_1, rep=param.BaseConfig.word_dimension, axis=1),
                shape=[-1, self.config.Y_maxlen, param.BaseConfig.word_dimension])

        with tf.variable_scope("fusion-layer-1"):
            self.x2_inter_1 = self.inter_rep_1 * self.input_X2
            self.fusion_output_1 = tf.concat(
                [self.input_X2, self.x2_inter_1, self.input_X2 - self.x2_inter_1, self.input_X2 * self.x2_inter_1,
                 self.x2_label],
                axis=-1)  # [Batch, len, 4 * dimension]
            self.fusion_output_1 = tf.layers.dense(inputs=self.fusion_output_1, units=self.config.mlp_output,
                                                   name='fusion-fnn')
            self.fusion_output_max_1 = tf.reduce_max(self.fusion_output_1, axis=-1)

        with tf.variable_scope("second-CNN-layer"):
            self.output_x1_2 = tf.layers.conv1d(self.output_x1_1,filters=self.config.filters_num,kernel_size=self.config.second_kernel_size,padding='same',name='second-cnn1')
            self.output_x2_2 = tf.layers.conv1d(self.output_x2_1,filters=self.config.filters_num,kernel_size=self.config.second_kernel_size,padding='same',name='second-cnn2')

        with tf.variable_scope("second-interaction"):
            self.inter_2 = self.interaction(self.output_x1_2,self.output_x2_2)
            self.inter_rep_2 = tf.reshape(tf.keras.backend.repeat_elements(self.inter_2, rep=param.BaseConfig.word_dimension, axis=1),shape=[-1,self.config.Y_maxlen,param.BaseConfig.word_dimension])

        with tf.variable_scope("fusion-layer-2"):
            self.x2_inter_2 = self.inter_rep_2 * self.input_X2
            self.fusion_output_2 = tf.concat(
                [self.input_X2, self.x2_inter_2, self.input_X2 - self.x2_inter_2, self.input_X2 * self.x2_inter_2, self.x2_label],
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
            self.inter_3 = self.interaction(self.output_x1_3,self.output_x2_3)
            self.inter_rep_3 = tf.reshape(tf.keras.backend.repeat_elements(self.inter_3, rep=param.BaseConfig.word_dimension, axis=1),shape=[-1,self.config.Y_maxlen,param.BaseConfig.word_dimension])

        with tf.variable_scope("fusion-layer-3"):
            self.x2_inter_3 = self.inter_rep_3 * self.input_X2
            self.fusion_output_3 = tf.concat([self.input_X2,self.x2_inter_3,self.input_X2 - self.x2_inter_3, self.input_X2 * self.x2_inter_3], axis=-1)  # [Batch, len, 2 + 4 * dimension]
            self.fusion_output_3 = tf.layers.dense(inputs=self.fusion_output_3,units=self.config.mlp_output,name='fusion-fnn')
            self.fusion_output_max_3 = tf.reduce_max(self.fusion_output_3,axis=-1) #[B,l]


        with tf.variable_scope("Augment-layer"):
            self.fusion_output_x2 = tf.concat([self.fusion_output_max_1,self.fusion_output_max_2,self.fusion_output_max_3],
                                            axis=-1) #[B,2l]
            self.fusion_output_x2 = tf.nn.relu(
                tf.layers.dense(inputs=self.fusion_output_x2, units=self.config.mlp_output, name='fnn1'))

            self.fusion_output_x1 =  tf.nn.relu(
                tf.layers.dense(inputs=self.new_x1_, units=self.config.mlp_output, name='fnn2'))

            self.fusion_output = tf.concat([self.fusion_output_x1,self.fusion_output_x2],axis=-1)

        # with tf.variable_scope("Augment-layer"):
        #     self.inter_3 = tf.concat([self.inter_2,self.inter_3],axis=-1)

        with tf.variable_scope("predict-layer"):
            self.output_1 = tf.nn.relu(tf.layers.dense(inputs=self.fusion_output,units=self.config.mlp_output,name='fnn1'))
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

    def gate1(self, ks, inputx):
        with tf.name_scope("gate"):
            weight_1 = tf.Variable(tf.random_normal([param.BaseConfig.word_dimension, 1],
                                                    stddev=0, seed=1), trainable=True, name='w1')
            weight_2 = tf.Variable(tf.random_normal([param.BaseConfig.word_dimension, 2],

                                                    stddev=0, seed=2), trainable=True, name='w2')
            weight_3 = tf.Variable(tf.random_normal([param.BaseConfig.word_dimension, 2],
                                                    stddev=0, seed=3), trainable=True, name='w3')

            k_1_init, k_2_init, k_3_init = ks[:, 0, :], ks[:, 1, :], ks[:, 2, :]  # [None,d]
            k_1 = tf.reshape(tf.keras.backend.repeat_elements(k_1_init, rep=self.config.X_maxlen, axis=1),
                             shape=[-1, self.config.X_maxlen, 1, param.BaseConfig.word_dimension])
            k_2 = tf.reshape(tf.keras.backend.repeat_elements(k_2_init, rep=self.config.X_maxlen, axis=1),
                             shape=[-1, self.config.X_maxlen, 1, param.BaseConfig.word_dimension])
            k_3 = tf.reshape(tf.keras.backend.repeat_elements(k_3_init, rep=self.config.X_maxlen, axis=1),
                             shape=[-1, self.config.X_maxlen, 1, param.BaseConfig.word_dimension])
            inputx_epd = tf.expand_dims(inputx, axis=2) #[b,l,1,d]
            fun1 = tf.einsum('abcd,de->abce', inputx_epd, weight_1)
            ksw_1 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun1, k_1)))  # [batch,l,1,d]

            fun2 = tf.einsum('abcd,de->abce', inputx_epd, weight_2)
            ksw_2 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun2, tf.concat([k_2,ksw_1],axis=2)))) # [batch,l,d]

            fun3 = tf.einsum('abcd,de->abce',inputx_epd , weight_3)
            ksw_3 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun3, tf.concat([k_3,ksw_2],axis=2)))) # [batch,l,d]

            n_vector_ = (ksw_1 + ksw_2 + ksw_3) * inputx_epd
            n_vector = tf.reshape(n_vector_, shape=[-1,self.config.X_maxlen,param.BaseConfig.word_dimension])

        return n_vector




