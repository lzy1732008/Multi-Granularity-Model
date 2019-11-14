import tensorflow as tf

def genericAttention(X,Y):
    with tf.variable_scope("generic attention"):
        dot_matrix = tf.matmul(X,tf.transpose(Y,[0,2,1]))
        x_2_y = tf.nn.softmax(dot_matrix, axis=2)  # x对y每个词的关注度
        y_2_x = tf.nn.softmax(dot_matrix, axis=1)  # y对x每个词的关注度
        return x_2_y, y_2_x
