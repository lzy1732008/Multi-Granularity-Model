#对比试验：融合层基于法条和事实同时使用，并且没有加入法条的前后件信息
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
        with tf.variable_scope("zero-interaction-layer"):
            self.inter_0, self.inter_x_0 = self.interaction(self.input_X1,self.input_X2)
            self.inter_rep_0 = tf.reshape(
                tf.keras.backend.repeat_elements(self.inter_0, rep=param.BaseConfig.word_dimension, axis=1),
                shape=[-1, self.config.Y_maxlen, param.BaseConfig.word_dimension])

            self.inter_rep_x_0 = tf.reshape(
                tf.keras.backend.repeat_elements(self.inter_x_0, rep=param.BaseConfig.word_dimension, axis=1),
                shape=[-1, self.config.X_maxlen, param.BaseConfig.word_dimension])

        with tf.variable_scope("fusion-layer-0"):
            self.x2_inter_0 = self.inter_rep_0 * self.input_X2
            self.fusion_output_0 = tf.concat(
                [self.input_X2, self.x2_inter_0, self.input_X2 - self.x2_inter_0, self.input_X2 * self.x2_inter_0],
                axis=-1)  # [Batch, len, 4 * dimension]
            self.fusion_output_0 = tf.layers.dense(inputs=self.fusion_output_0, units=self.config.mlp_output,
                                                   name='fusion-fnn')

            self.fusion_output_max_0 = tf.reduce_max(self.fusion_output_0, axis=-1)

        with tf.variable_scope("fusion-layer-x-0"):
            self.x1_inter_0 = self.inter_rep_x_0 * self.input_X1
            self.fusion_output_x_0 = tf.concat(
                [self.input_X1, self.x1_inter_0, self.input_X1 - self.x1_inter_0, self.input_X1 * self.x1_inter_0],
                axis=-1)
            self.fusion_output_x_0 = tf.layers.dense(inputs=self.fusion_output_x_0, units=self.config.mlp_output,name='fusion-fnn')
            self.fusion_output_max_x_0 = tf.reduce_max(self.fusion_output_x_0, axis=-1)

        with tf.variable_scope("first-CNN-layer"):
            self.output_x1_1 = tf.layers.conv1d(self.input_X1,filters=self.config.filters_num,kernel_size=self.config.first_kernel_size,padding='same',name='first-cnn1')
            self.output_x2_1 = tf.layers.conv1d(self.input_X2,filters=self.config.filters_num,kernel_size=self.config.first_kernel_size,padding='same',name='first-cnn2')

        with tf.variable_scope("first-interaction"):
            self.inter_1, self.inter_x_1 = self.interaction(self.output_x1_1, self.output_x2_1)

            self.inter_rep_1 = tf.reshape(
                tf.keras.backend.repeat_elements(self.inter_1, rep=param.BaseConfig.word_dimension, axis=1),
                shape=[-1, self.config.Y_maxlen, param.BaseConfig.word_dimension])

            self.inter_rep_x_1 = tf.reshape(
                tf.keras.backend.repeat_elements(self.inter_x_1, rep=param.BaseConfig.word_dimension, axis=1),
                shape=[-1, self.config.X_maxlen, param.BaseConfig.word_dimension])

        with tf.variable_scope("fusion-layer-1"):
            self.x2_inter_1 = self.inter_rep_1 * self.input_X2
            self.fusion_output_1 = tf.concat(
                [self.input_X2, self.x2_inter_1, self.input_X2 - self.x2_inter_1, self.input_X2 * self.x2_inter_1],
                axis=-1)  # [Batch, len, 4 * dimension]
            self.fusion_output_1 = tf.layers.dense(inputs=self.fusion_output_1, units=self.config.mlp_output,
                                                   name='fusion-fnn')
            # self.fusion_output_1 = tf.layers.dropout(self.fusion_output_1,rate=self.dropout_rate)
            self.fusion_output_max_1 = tf.reduce_max(self.fusion_output_1, axis=-1)

        with tf.variable_scope("fusion-layer-x-1"):
            self.x1_inter_1 = self.inter_rep_x_1 * self.input_X1
            self.fusion_output_x_1 = tf.concat(
                [self.input_X1, self.x1_inter_1, self.input_X1 - self.x1_inter_1, self.input_X1 * self.x1_inter_1],
                axis=-1)
            self.fusion_output_x_1 = tf.layers.dense(inputs=self.fusion_output_x_1, units=self.config.mlp_output,name='fusion-fnn')
            self.fusion_output_max_x_1 = tf.reduce_max(self.fusion_output_x_1, axis=-1)


        with tf.variable_scope("second-CNN-layer"):
            self.output_x1_2 = tf.layers.conv1d(self.output_x1_1,filters=self.config.filters_num,kernel_size=self.config.second_kernel_size,padding='same',name='second-cnn1')
            self.output_x2_2 = tf.layers.conv1d(self.output_x2_1,filters=self.config.filters_num,kernel_size=self.config.second_kernel_size,padding='same',name='second-cnn2')

        with tf.variable_scope("second-interaction"):
            self.inter_2, self.inter_x_2 = self.interaction(self.output_x1_2,self.output_x2_2)
            self.inter_rep_2 = tf.reshape(tf.keras.backend.repeat_elements(self.inter_2, rep=param.BaseConfig.word_dimension, axis=1),shape=[-1,self.config.Y_maxlen,param.BaseConfig.word_dimension])
            self.inter_rep_x_2 = tf.reshape(
                tf.keras.backend.repeat_elements(self.inter_x_2, rep=param.BaseConfig.word_dimension, axis=1),
                shape=[-1, self.config.X_maxlen, param.BaseConfig.word_dimension])


        with tf.variable_scope("fusion-layer-2"):
            self.x2_inter_2 = self.inter_rep_2 * self.input_X2
            self.fusion_output_2 = tf.concat(
                [self.input_X2, self.x2_inter_2, self.input_X2 - self.x2_inter_2, self.input_X2 * self.x2_inter_2],
                axis=-1)  # [Batch, len, 4 * dimension]
            self.fusion_output_2 = tf.layers.dense(inputs=self.fusion_output_2, units=self.config.mlp_output,
                                                   name='fusion-fnn')
            self.fusion_output_max_2 = tf.reduce_max(self.fusion_output_2, axis=-1)

        with tf.variable_scope("fusion-layer-x-2"):
            self.x1_inter_2 = self.inter_rep_x_2 * self.input_X1
            self.fusion_output_x_2 = tf.concat(
                [self.input_X1, self.x1_inter_2, self.input_X1 - self.x1_inter_2, self.input_X1 * self.x1_inter_2],
                axis=-1)
            self.fusion_output_x_2 = tf.layers.dense(inputs=self.fusion_output_x_2, units=self.config.mlp_output,name='fusion-fnn')
            self.fusion_output_max_x_2 = tf.reduce_max(self.fusion_output_x_2, axis=-1)

        with tf.variable_scope("third-CNN-layer"):
            self.output_x1_3 = tf.layers.conv1d(self.output_x1_2, filters=self.config.filters_num,
                                                kernel_size=self.config.third_kernel_size, padding='same',
                                                name='second-cnn1')
            self.output_x2_3 = tf.layers.conv1d(self.output_x2_2, filters=self.config.filters_num,
                                                kernel_size=self.config.third_kernel_size, padding='same',
                                                name='second-cnn2')
        with tf.variable_scope("third-interaction"):
            self.inter_3, self.inter_x_3 = self.interaction(self.output_x1_3,self.output_x2_3)
            self.inter_rep_3 = tf.reshape(tf.keras.backend.repeat_elements(self.inter_3, rep=param.BaseConfig.word_dimension, axis=1),shape=[-1,self.config.Y_maxlen,param.BaseConfig.word_dimension])
            self.inter_rep_x_3 = tf.reshape(
                tf.keras.backend.repeat_elements(self.inter_x_3, rep=param.BaseConfig.word_dimension, axis=1),
                shape=[-1, self.config.X_maxlen, param.BaseConfig.word_dimension])

        with tf.variable_scope("fusion-layer-3"):
            self.x2_inter_3 = self.inter_rep_3 * self.input_X2
            self.fusion_output_3 = tf.concat([self.input_X2,self.x2_inter_3,self.input_X2 - self.x2_inter_3, self.input_X2 * self.x2_inter_3], axis=-1)  # [Batch, len, 2 + 4 * dimension]
            self.fusion_output_3 = tf.layers.dense(inputs=self.fusion_output_3,units=self.config.mlp_output,name='fusion-fnn')
            self.fusion_output_max_3 = tf.reduce_max(self.fusion_output_3,axis=-1) #[B,l]

        with tf.variable_scope("fusion-layer-x-3"):
            self.x1_inter_3 = self.inter_rep_x_3 * self.input_X1
            self.fusion_output_x_2 = tf.concat(
                [self.input_X1, self.x1_inter_3, self.input_X1 - self.x1_inter_3, self.input_X1 * self.x1_inter_3],
                axis=-1)
            self.fusion_output_x_3 = tf.layers.dense(inputs=self.fusion_output_x_3, units=self.config.mlp_output,name='fusion-fnn')
            self.fusion_output_max_x_3 = tf.reduce_max(self.fusion_output_x_3, axis=-1)

        with tf.variable_scope("Augment-layer"):
            self.fusion_output = tf.concat([self.fusion_output_max_0, self.fusion_output_max_1,self.fusion_output_max_2,self.fusion_output_max_3],
                                            axis=-1) #[B,2l]

            self.fusion_output_x = tf.concat([self.fusion_output_max_x_0, self.fusion_output_max_x_1, self.fusion_output_max_x_2, self.fusion_output_max_x_3]
                                             ,axis=-1)

        with tf.variable_scope("predict-layer"):
            self.output_1 = tf.nn.relu(tf.layers.dense(inputs=self.fusion_output,units=self.config.mlp_output,name='fnn1'))
            self.output_1 = tf.layers.dropout(self.output_1,rate=self.dropout_rate)

            self.output_x_1 =  tf.nn.relu(tf.layers.dense(inputs=self.fusion_output_x,units=self.config.mlp_output,name='fnn1x'))
            self.output_x_1 = tf.layers.dropout(self.output_x_1, rate=self.dropout_rate)

            final_input = tf.concat([self.output_1, self.output_x_1],axis=-1)

            self.logit = tf.layers.dense(inputs=final_input,units=2,name='fnn2')

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

        #计算事实的总权重
        x2_weight_ = tf.reduce_mean(x1_2_x2,axis=1)  #[Batch,len2]

        x2_weight_exp = tf.expand_dims(x2_weight_,axis=2) #[Batch, len2, 1]
        x1_weight_exp = tf.matmul(x2_weight_exp, x2_2_x1) #[Batch, len, 1]
        x1_weight_output = tf.reshape(x1_weight_exp, shape=[-1, len1])

        return x2_weight, x1_weight_output





