import tensorflow as tf
import models.parameter as param
import models.modules as modules


class MultiGraConfig:
    def __init__(self):
        # v1
        self.X_maxlen = 30
        self.Y_maxlen = 50
        self.dropout_rate = 0.8
        self.first_kernel_size = 2
        self.second_kernel_size = 4
        self.third_kernel_size = 8
        self.filters_num = param.BaseConfig.word_dimension
        self.mlp_output = 2 * self.Y_maxlen
        self.knowledge_dimension = param.BaseConfig.word_dimension


class MultiGranularityCNNModel:
    def __init__(self, config):
        self.config = config
        self.input_X1 = tf.placeholder(name="inputX1_word", dtype=tf.float32,
                                       shape=[None, self.config.X_maxlen, param.BaseConfig.word_dimension])
        self.input_X2 = tf.placeholder(name="inputX2_word", dtype=tf.float32,
                                       shape=[None, self.config.Y_maxlen, param.BaseConfig.word_dimension])
        self.x2_label = tf.placeholder(name="inputX2_label", dtype=tf.int32,
                                       shape=[None, self.config.Y_maxlen])
        self.y = tf.placeholder(name="target_y", dtype=tf.int32, shape=[None, 2])

        self.dropout_rate = tf.placeholder(tf.float32, name='keep_prob')

        self.build_model()

    def build_model(self):
        with tf.variable_scope("encoder-layer"):
            knowledge_embedding = tf.Variable(tf.random_normal([3, self.config.knowledge_dimension],
                                                               stddev=0, seed=1), trainable=False,
                                                               name='knowledge-embedding')
            self.x2_label_embedding = tf.nn.embedding_lookup(knowledge_embedding, self.x2_label)
            KG = modules.KnowledgeGate(2, self.input_X2, self.x2_label_embedding)
            self.encoder_x2 = KG.exeKnowledgeGate()


        with tf.variable_scope("first-CNN-layer"):
            self.output_x1_1 = tf.layers.conv1d(self.input_X1, filters=self.config.filters_num,
                                                kernel_size=self.config.first_kernel_size, padding='same',
                                                name='first-cnn1',
                                                activation=tf.nn.relu)
            self.output_x2_1 = tf.layers.conv1d(self.encoder_x2, filters=self.config.filters_num,
                                                kernel_size=self.config.first_kernel_size, padding='same',
                                                name='first-cnn2',
                                                activation=tf.nn.relu)

        with tf.variable_scope("first-interaction"):
            interaction = modules.Interaction(4, self.output_x1_1, self.output_x2_1)
            self.inter1_output_x2 = interaction.exeInteraction()

        with tf.variable_scope("first-fusion"):
            interaction = modules.Interaction(9, self.output_x1_1, self.output_x2_1)
            self.fusion_output_11, self.fusion_output_12 = interaction.exeInteraction()

        with tf.variable_scope("second-CNN-layer"):
            self.output_x1_2 = tf.layers.conv1d(self.fusion_output_11, filters=self.config.filters_num,
                                                kernel_size=self.config.second_kernel_size, padding='same',
                                                name='second-cnn1',
                                                activation=tf.nn.relu)
            self.output_x2_2 = tf.layers.conv1d(self.fusion_output_12, filters=self.config.filters_num,
                                                kernel_size=self.config.second_kernel_size, padding='same',
                                                name='second-cnn2',
                                                activation=tf.nn.relu)

        with tf.variable_scope("second-interaction"):
            interaction = modules.Interaction(4, self.output_x1_2, self.output_x2_2)
            self.inter2_output_x2 = interaction.exeInteraction()

        with tf.variable_scope("second-fusion"):
            interaction = modules.Interaction(9, self.output_x1_2, self.output_x2_2)
            self.fusion_output_21,self.fusion_output_22 = interaction.exeInteraction()

        with tf.variable_scope("third-CNN-layer"):
            self.output_x1_3 = tf.layers.conv1d(self.fusion_output_21, filters=self.config.filters_num,
                                                kernel_size=self.config.third_kernel_size, padding='same',
                                                name='third-cnn1',
                                                activation=tf.nn.relu)
            self.output_x2_3 = tf.layers.conv1d(self.fusion_output_22, filters=self.config.filters_num,
                                                kernel_size=self.config.third_kernel_size, padding='same',
                                                name='third-cnn2',
                                                activation=tf.nn.relu)

        with tf.variable_scope("third-interaction"):
            interaction = modules.Interaction(4, self.output_x1_3, self.output_x2_3)
            self.inter3_output_x2 = interaction.exeInteraction()

        with tf.variable_scope("combine-layer"):
            self.beta1 = tf.Variable(tf.random_normal([1], stddev=0, seed=1), trainable=True, name='inter-weight-1')
            self.beta2 = tf.Variable(tf.random_normal([1], stddev=0, seed=1), trainable=True, name='inter-weight-2')
            self.beta3 = tf.Variable(tf.random_normal([1], stddev=0, seed=1), trainable=True, name='inter-weight-3')
            self.combine = tf.concat([self.beta1[0] * self.inter1_output_x2,self.beta2[0] * self.inter2_output_x2, self.beta3[0] * self.inter3_output_x2],axis=-1)

        with tf.variable_scope("predict-layer"):
            self.output_ = tf.nn.relu(
                tf.layers.dense(inputs=self.combine, units=self.config.mlp_output, name='fnn1'))
            self.output_ = tf.layers.dropout(self.output_, rate=self.dropout_rate)
            self.logit = tf.layers.dense(inputs=self.output_, units=2, name='fnn2')

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

