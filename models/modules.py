import tensorflow as tf
import models.components as comp

class Interaction:
    def __init__(self, method, *data):
        self.method = method
        self.data = data

    def exeInteraction(self):
        if self.method == 1:
            return self.playInteraction1()

    def playInteraction1(self):
        '''
        该interaction包含两个输入
        过程如下：
        1.计算普通的attention的矩阵
        2.然后分别对行求softmax和列softmax
        3.计算两个输入各自被对齐后的向量矩阵
        '''
        with tf.variable_scope("interactin-layer"):
            assert len(self.data) == 2, ValueError('the number of input data is wrong, it should be 2,but{0}'.format(len(self.data)))
            #首先计算attention
            x_2_y, y_2_x = comp.genericAttention(self.data[0],self.data[1])

            #获取x和y的各自总权重
            x_weight = tf.reduce_sum(y_2_x,axis=-1)
            y_weight = tf.reduce_sum(x_2_y,axis=1)

            #计算对齐后的向量矩阵
            new_x = tf.einsum('abc,ab->abc',self.data[0],x_weight)
            new_y = tf.einsum('abc,ab->abc',self.data[1],y_weight)

            return new_x, new_y

class Fusion:
    def __init__(self, method, *data):
        self.method = method
        self.data = data

    def exeFusion(self):
        if self.method == 1: #just concat them
            return self.playFusion1()

    def playFusion1(self):
        assert len(self.data) == 2, ValueError(
            'the number of input data is wrong, it should be 2,but{0}'.format(len(self.data)))
        with tf.variable_scope("fusion-layer"):
            return tf.concat(self.data,axis=-1)
