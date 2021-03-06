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
            elif self.method == 4:
                return self.playInteraction4()
            elif self.method == 5:
                return self.playInteraction5()
            elif self.method == 6:
                return self.playInteraction6()
            elif self.method == 7:
                return self.playInteraction7()
            elif self.method == 8:
                return self.playInteraction8()
            elif self.method == 9:
                return self.playInteraction9()
            elif self.method == 10:
                return self.playInteraction10()
            elif self.method == 11:
                return self.playInteraction11()
            elif self.method == 12:
                return self.playInteraction12()
            elif self.method == 13:
                return self.playInteraction13()
            elif self.method == 14:
                return self.playInteraction14()


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


    def playInteraction7(self):
        assert len(self.data) == 2, ValueError(
            'the number of input data is wrong, it should be 2,but{0}'.format(len(self.data)))
        # 首先计算attention
        x_2_y, y_2_x = comp.genericAttention(self.data[0], self.data[1])
        feature_b = tf.matmul(y_2_x, self.data[0], transpose_a=True)
        feature_a = tf.matmul(x_2_y, self.data[1])
        return feature_a, feature_b

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

    def playInteraction4(self):
        inputX1 = self.data[0]
        inputX2 = self.data[1]
        len1 = inputX1.get_shape().as_list()[1]
        len2 = inputX2.get_shape().as_list()[1]
        x1_2_x2,x2_2_x1 = comp.genericAttention(self.data[0], self.data[1])

        # 计算x1每个词获取的总weight
        x1_weight = tf.reduce_mean(x2_2_x1, axis=2)  # [Batch, len1]

        # 计算x2最后获取的每个词的总的weight
        x1_weight_ = tf.expand_dims(x1_weight, axis=1)  # [Batch, 1, len1]
        x2_weight = tf.matmul(x1_weight_, x1_2_x2)  # [Batch, 1, len2]
        x2_weight = tf.reshape(x2_weight, shape=[-1, len2])
        return x2_weight

    def playInteraction5(self):
        x1_len = self.data[0].get_shape().as_list()[1]
        x2_len = self.data[1].get_shape().as_list()[1]
        ks_len = self.data[2].get_shape().as_list()[1]

        assert ks_len == x2_len,ValueError("ks:{0}, law:{1}".format(ks_len,x2_len))

        beta = self.data[-1]
        ks_rep = tf.reshape(tf.keras.backend.repeat_elements(self.data[2],rep=x1_len,axis=1),shape=[-1,x1_len,x2_len])
        dot_matrix = tf.matmul(self.data[0], tf.transpose(self.data[1], [0, 2, 1])) + beta[0] * ks_rep #[None, m, n]
        x_2_y = tf.nn.softmax(dot_matrix, axis=2)  # x对y每个词的关注度
        y_2_x = tf.nn.softmax(dot_matrix, axis=1)  # y对x每个词的关注度

        # 计算x1每个词获取的总weight
        x1_weight = tf.reduce_mean(y_2_x, axis=2)  # [Batch, len1]

        # 计算x2最后获取的每个词的总的weight
        x1_weight_ = tf.expand_dims(x1_weight, axis=1)  # [Batch, 1, len1]
        x2_weight = tf.matmul(x1_weight_, x_2_y)  # [Batch, 1, len2]
        x2_weight = tf.reshape(x2_weight, shape=[-1, x2_len])
        return x2_weight

    def playInteraction6(self):
        x1_len = self.data[0].get_shape().as_list()[1]
        x2_len = self.data[1].get_shape().as_list()[1]
        ks_len = self.data[2].get_shape().as_list()[1]

        assert ks_len == x2_len,ValueError("ks:{0}, law:{1}".format(ks_len,x2_len))

        beta = self.data[-1]
        ks_rep = tf.reshape(tf.einsum('abc,cd->abd',self.data[2],beta),shape=[-1,x2_len] )#[B,x2_len,1]
        ks_rep = tf.reshape(tf.keras.backend.repeat_elements(ks_rep,rep=x1_len,axis=1),shape=[-1,x1_len,x2_len])
        dot_matrix = tf.matmul(self.data[0], tf.transpose(self.data[1], [0, 2, 1])) +  ks_rep #[None, m, n]
        x_2_y = tf.nn.softmax(dot_matrix, axis=2)  # x对y每个词的关注度
        y_2_x = tf.nn.softmax(dot_matrix, axis=1)  # y对x每个词的关注度

        # 计算x1每个词获取的总weight
        x1_weight = tf.reduce_mean(y_2_x, axis=2)  # [Batch, len1]

        # 计算x2最后获取的每个词的总的weight
        x1_weight_ = tf.expand_dims(x1_weight, axis=1)  # [Batch, 1, len1]
        x2_weight = tf.matmul(x1_weight_, x_2_y)  # [Batch, 1, len2]
        x2_weight = tf.reshape(x2_weight, shape=[-1, x2_len])
        return x2_weight

    def playInteraction8(self):
        x1_len = self.data[0].get_shape().as_list()[1]
        x2_len,dim = self.data[1].get_shape().as_list()[1:]
        ks_len = self.data[2].get_shape().as_list()[1]

        assert ks_len == x2_len, ValueError("ks:{0}, law:{1}".format(ks_len, x2_len))
        beta = tf.Variable(tf.random_normal(shape=[dim, 2], stddev=0, seed=1, dtype=tf.float32), trainable=True,
                                 name='beta')
        weigt = tf.einsum('abc,cd->abd',self.data[1],beta) #[B,l2,2]
        ks = tf.reduce_sum(weigt * self.data[2],axis=-1) #[B,l2]
        ks_rep = tf.reshape(tf.keras.backend.repeat_elements(ks, rep=x1_len, axis=1), shape=[-1, x1_len, x2_len])
        dot_matrix = tf.matmul(self.data[0],self.data[1],transpose_b=True) + ks_rep
        x_2_y = tf.nn.softmax(dot_matrix, axis=2)  # x对y每个词的关注度
        y_2_x = tf.nn.softmax(dot_matrix, axis=1)  # y对x每个词的关注度

        # 计算x1每个词获取的总weight
        x1_weight = tf.reduce_mean(y_2_x, axis=2)  # [Batch, len1]

        # 计算x2最后获取的每个词的总的weight
        x1_weight_ = tf.expand_dims(x1_weight, axis=1)  # [Batch, 1, len1]
        x2_weight = tf.matmul(x1_weight_, x_2_y)  # [Batch, 1, len2]
        x2_weight = tf.reshape(x2_weight, shape=[-1, x2_len])
        return x2_weight

    def playInteraction9(self):
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
        x_weight = tf.reduce_mean(y_2_x, axis=-1)
        y_weight = tf.reduce_mean(x_2_y, axis=1)

        # 计算对齐后的向量矩阵
        new_x = tf.einsum('abc,ab->abc', self.data[0], x_weight)
        new_y = tf.einsum('abc,ab->abc', self.data[1], y_weight)

        return new_x, new_y

    def playInteraction10(self):
        x1_len = self.data[0].get_shape().as_list()[1]
        x2_len,dim = self.data[1].get_shape().as_list()[1:]
        ks_len = self.data[2].get_shape().as_list()[1]

        assert ks_len == x2_len, ValueError("ks:{0}, law:{1}".format(ks_len, x2_len))

        ks = tf.reduce_mean(self.data[1] * self.data[2],axis=-1)
        ks_rep = tf.reshape(tf.keras.backend.repeat_elements(ks, rep=x1_len, axis=1), shape=[-1, x1_len, x2_len])
        dot_matrix = tf.matmul(self.data[0],self.data[1],transpose_b=True) + ks_rep
        x_2_y = tf.nn.softmax(dot_matrix, axis=2)  # x对y每个词的关注度
        y_2_x = tf.nn.softmax(dot_matrix, axis=1)  # y对x每个词的关注度

        # 计算x1每个词获取的总weight
        x1_weight = tf.reduce_mean(y_2_x, axis=2)  # [Batch, len1]

        # 计算x2最后获取的每个词的总的weight
        x1_weight_ = tf.expand_dims(x1_weight, axis=1)  # [Batch, 1, len1]
        x2_weight = tf.matmul(x1_weight_, x_2_y)  # [Batch, 1, len2]
        x2_weight = tf.reshape(x2_weight, shape=[-1, x2_len])
        return x2_weight

    def playInteraction11(self):
        '''
        输入：两个
        过程：
        1.执行playInteraction1获取每个的新表达
        :return: new_x,new_y
        '''
        y_len = self.data[1].get_shape().as_list()[1]

        x_2_y, y_2_x = comp.genericAttention(self.data[0], self.data[1])

        # 获取x和y的各自总权重
        x_weight = tf.reduce_sum(y_2_x, axis=-1)
        y_weight = tf.reduce_sum(x_2_y, axis=1)

        # 计算对齐后的向量矩阵
        new_x = tf.einsum('abc,ab->abc', self.data[0], x_weight)
        new_y = tf.einsum('abc,ab->abc', self.data[1], y_weight)
        return new_x, new_y

    def playInteraction12(self):
        x2_len,dim = self.data[1].get_shape().as_list()[1:]

        dot_matrix = tf.matmul(self.data[0],self.data[1],transpose_b=True) + self.data[2]
        x_2_y = tf.nn.softmax(dot_matrix, axis=2)  # x对y每个词的关注度
        y_2_x = tf.nn.softmax(dot_matrix, axis=1)  # y对x每个词的关注度

        # 计算x1每个词获取的总weight
        x1_weight = tf.reduce_mean(y_2_x, axis=2)  # [Batch, len1]

        # 计算x2最后获取的每个词的总的weight
        x1_weight_ = tf.expand_dims(x1_weight, axis=1)  # [Batch, 1, len1]
        x2_weight = tf.matmul(x1_weight_, x_2_y)  # [Batch, 1, len2]
        x2_weight = tf.reshape(x2_weight, shape=[-1, x2_len])
        return x2_weight

    def playInteraction13(self):
        x2_len,dim = self.data[1].get_shape().as_list()[1:]
        alpha = tf.Variable(tf.random_normal(shape=[1], stddev=0, seed=1, dtype=tf.float32), trainable=True,
                           name='alpha')
        dot_matrix = tf.matmul(self.data[0],self.data[1],transpose_b=True) + alpha[0] * self.data[2]
        x_2_y = tf.nn.softmax(dot_matrix, axis=2)  # x对y每个词的关注度
        y_2_x = tf.nn.softmax(dot_matrix, axis=1)  # y对x每个词的关注度

        # 计算x1每个词获取的总weight
        x1_weight = tf.reduce_mean(y_2_x, axis=2)  # [Batch, len1]

        # 计算x2最后获取的每个词的总的weight
        x1_weight_ = tf.expand_dims(x1_weight, axis=1)  # [Batch, 1, len1]
        x2_weight = tf.matmul(x1_weight_, x_2_y)  # [Batch, 1, len2]
        x2_weight = tf.reshape(x2_weight, shape=[-1, x2_len])
        return x2_weight

    def playInteraction14(self):
        x2_len, dim = self.data[1].get_shape().as_list()[1:]
        dot_matrix = tf.matmul(self.data[0], self.data[1], transpose_b=True) * self.data[2]
        x_2_y = tf.nn.softmax(dot_matrix, axis=2)  # x对y每个词的关注度
        y_2_x = tf.nn.softmax(dot_matrix, axis=1)  # y对x每个词的关注度

        # 计算x1每个词获取的总weight
        x1_weight = tf.reduce_mean(y_2_x, axis=2)  # [Batch, len1]

        # 计算x2最后获取的每个词的总的weight
        x1_weight_ = tf.expand_dims(x1_weight, axis=1)  # [Batch, 1, len1]
        x2_weight = tf.matmul(x1_weight_, x_2_y)  # [Batch, 1, len2]
        x2_weight = tf.reshape(x2_weight, shape=[-1, x2_len])
        return x2_weight




class Fusion:
    def __init__(self, method, *data):
        self.method = method
        self.data = data

    def exeFusion(self):
        with tf.variable_scope("fusion-layer"):
            if self.method == 1:  # just concat them
                return self.playFusion1()


    def playFusion1(self):
        return tf.concat(self.data,axis=-1)


class KnowledgeGate:
    def __init__(self, method, data, knowledge):
        self.method = method
        self.data = data
        self.knowledge = knowledge

    def exeKnowledgeGate(self):
        with tf.variable_scope("knowledge-layer"):
            if self.method == 1:
                return self.playKG1()
            elif self.method == 2:
                return self.playKG2()


    def playKG1(self):
        k_dimension = self.knowledge.get_shape().as_list()[-1]
        data_dimension = self.data.get_shape().as_list()[-1]
        weight_1 = tf.Variable(tf.random_normal([k_dimension, data_dimension],
                                                stddev=0, seed=1), trainable=True, name='w1')
        weight_2 = tf.Variable(tf.random_normal([data_dimension, data_dimension],
                                                stddev=0, seed=1), trainable=True, name='w2')
        pw = tf.nn.sigmoid(
            tf.einsum('abc,cd->abd', self.knowledge, weight_1) + tf.einsum('abc,cd->abd', self.data, weight_2))
        output = (1 - pw) * self.data + pw * self.knowledge
        return output

    def playKG2(self):
        k_dimension = self.knowledge.get_shape().as_list()[-1]
        data_dimension = self.data.get_shape().as_list()[-1]
        weight = tf.Variable(tf.random_normal([data_dimension, k_dimension],
                                                stddev=0, seed=1), trainable=True, name='w1')
        pw = tf.sigmoid(tf.einsum('abc,cd->abd', self.data, weight) * self.knowledge)    # [B,l,d]
        output = pw * self.data
        return output







