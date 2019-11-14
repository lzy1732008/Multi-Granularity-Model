import tensorflow as tf
import models.components as comp

class Interaction:
    def __init__(self, method, *data):
        self.method = method
        self.data = data

    def exeInteraction(self):
        with tf.variable_scope("interaction-play"):
            if self.method == 1:
                return self.playInteraction1()
            elif self.method == 2:
                return self.playInteraction2()
            elif self.method == 3:
                return self.playInteraction3()


    def playInteraction1(self):
        '''
        该interaction包含两个输入
        过程如下：
        1.计算普通的attention的矩阵
        2.然后分别对行求softmax和列softmax
        3.计算两个输入各自被对齐后的向量矩阵
        '''
        assert len(self.data) == 2, ValueError(
            'the number of input data is wrong, it should be 2,but{0}'.format(len(self.data)))
        # 首先计算attention
        x_2_y, y_2_x = comp.genericAttention(self.data[0], self.data[1])

        # 获取x和y的各自总权重
        x_weight = tf.reduce_sum(y_2_x, axis=-1)
        y_weight = tf.reduce_sum(x_2_y, axis=1)

        # 计算对齐后的向量矩阵
        new_x = tf.einsum('abc,ab->abc', self.data[0], x_weight)
        new_y = tf.einsum('abc,ab->abc', self.data[1], y_weight)

        return new_x, new_y


    def playInteraction2(self):
        '''
        该interaction过程如下
        输入：两个
        过程：
        1. 计算普通的attention矩阵
        2. 然后分别对行求softmax和列softmax
        3. 分别对行和列做加和，获取x和y的权重
        :return: x_weight, y_weight
        '''
        x_2_y, y_2_x = comp.genericAttention(self.data[0], self.data[1])

        # 获取x和y的各自总权重
        x_weight = tf.reduce_sum(y_2_x, axis=-1)
        y_weight = tf.reduce_sum(x_2_y, axis=1)
        return x_weight, y_weight


    def playInteraction3(self):
        '''
        输入：两个
        过程：
        1.执行playInteraction1获取每个的新表达
        2.执行attention-over-attention获取其中一个的权重
        :return: new_x,new_y,y_weight
        '''
        y_len = self.data[1].get_shape().as_list()[1]

        x_2_y, y_2_x = comp.genericAttention(self.data[0], self.data[1])

        # 获取x和y的各自总权重
        x_weight = tf.reduce_sum(y_2_x, axis=-1)
        y_weight = tf.reduce_sum(x_2_y, axis=1)

        # 计算对齐后的向量矩阵
        new_x = tf.einsum('abc,ab->abc', self.data[0], x_weight)
        new_y = tf.einsum('abc,ab->abc', self.data[1], y_weight)

        #计算Y的权重
        x_weight_ = tf.expand_dims(x_weight,axis=1)
        y_weight_ = tf.matmul(x_weight_,x_2_y) #[Batch, 1, len2]
        y_weight_output = tf.reshape(y_weight_,shape=[-1,y_len])
        return new_x, new_y,y_weight_output


class Fusion:
    def __init__(self, method, *data):
        self.method = method
        self.data = data

    def exeFusion(self):
        if self.method == 1: #just concat them
            return self.playFusion1()

    def playFusion1(self):
        with tf.variable_scope("fusion-layer"):
            return tf.concat(self.data,axis=-1)
