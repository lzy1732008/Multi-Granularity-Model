import tensorflow as tf
import models.parameter as param

class Config:
    def __init__(self):

        #param v1
        self.max_len = 70 #这个有待统计
        self.rnn_output_dim = 128
        self.fnn_output = 64
        self.num_layers = 1
        self.dropout_rate = 0.8


class QHModel:
    def __init__(self):
        self.config = Config()
        self.inputX = tf.placeholder(dtype=tf.float32,shape=[None,None,param.BaseConfig.law_word_dimension],name="inputX")
        self.inputSplit = tf.placeholder(dtype=tf.int32,shape=[None,3],name="inputSplit") #其中每行以第一个元素是bi-lstm 前向切分点，第二项是后向切分点
        self.y = tf.placeholder(dtype=tf.int32,shape=[None,3],name="inputY")
        self.dropout_rate = tf.placeholder(dtype=tf.float32,name="dropout_rate")

        self.run_model()


    def lstm_cell(self):   # lstm核
        return tf.contrib.rnn.BasicLSTMCell(self.config.rnn_output_dim, forget_bias= 1.0, state_is_tuple=True)

    def dropout(self): # 为每一个rnn核后面加一个dropout层
        return tf.contrib.rnn.DropoutWrapper(self.lstm_cell(), output_keep_prob=self.dropout_rate)

    def run_model(self):
        with tf.variable_scope("bi-lstm-layer"):
            # 首先输入到一层bi-lstm中
            cell_fw = [self.dropout() for _ in range(self.config.num_layers)]
            cell_bw = [self.dropout() for _ in range(self.config.num_layers)]

            cell_fw = tf.contrib.rnn.MultiRNNCell(cell_fw, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.MultiRNNCell(cell_bw, state_is_tuple=True)

            init_state_fw = cell_fw.zero_state(batch_size=param.BaseConfig.batch_size, dtype=tf.float32)
            init_state_bw = cell_bw.zero_state(batch_size=param.BaseConfig.batch_size, dtype=tf.float32)

            _outputs_1, state_1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                                  inputs=self.inputX,
                                                                  initial_state_fw=init_state_fw,
                                                                  initial_state_bw=init_state_bw)
            output_fw = _outputs_1[0]  # shape:[batch_size,max_time,output_size]
            output_bw = _outputs_1[1]

        with tf.variable_scope("split-layer"):
            #获取分割点的输出
            preCut = self.inputSplit[:,:2]
            hCut = tf.reshape(tf.concat([self.inputSplit[:,0],self.inputSplit[:,2]],axis=-1),shape=[-1, 2])
            output_split_fw = tf.gather_nd(output_fw,indices=preCut) #[batch, d]
            output_split_bw = tf.gather_nd(output_bw,indices=hCut) #[batch, d]
            output_split = tf.reshape(tf.concat([output_split_fw,output_split_bw],axis=-1),shape=[-1, param.BaseConfig.word_dimension * 2]) #[batch, 2*d]

        with tf.variable_scope("predict-layer"):
            self.output_ = tf.nn.relu(tf.layers.dense(inputs=output_split,units=self.config.fnn_output,name='fnn1'))
            self.output_ = tf.layers.dropout(self.output_,rate=self.dropout_rate)
            self.logit = tf.layers.dense(inputs=self.output_,units=3,name='fnn2')

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




