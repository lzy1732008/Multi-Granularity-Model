# import re
#
# # pattern = '，|。|；|：'
# # regx = re.compile(pattern)
# # input = "鼎折覆餗。asdhfjsdhfk；"
# # print(regx.split(input))
#
# import tensorflow as tf
# inputArray2 = tf.convert_to_tensor([[1.0,1.2,3.0],[2.1,2.2,3.2],[3.3,3.4,4.3],[0.3,0.4,1.4]])
# inputArray1 = tf.convert_to_tensor([[9.0,10.2,11.0],[10.3,10.4,11.4],[11.5,11.6,12.6],[15.5,15.6,16.6]])
# # splitArray = tf.convert_to_tensor([[0,1,2],[1,1,2]])
# # gather_output = tf.gather_nd(inputArray,indices=splitArray[:,:2])
# # concat = tf.stack([inputArray1,inputArray2],axis=-1)
#


#统计0，1分类0.7966
# import json
# fr = open('resource/lhjf_ft_labeled.json','r',encoding='utf-8')
# allft = json.load(fr).items()
# alldata = []
# num_0 = 0
# num_1 = 0
# num_2 = 0
# for ft,labeledcontents in allft:
#     for i, c in enumerate(labeledcontents):
#         if int(c[1]) == 0: num_0 += 1
#         elif int(c[1]) == 1: num_1 += 1
#         elif int(c[1]) == 2: num_2 += 1
#
# print("0:{0},1:{1}，2:{2}".format(num_0,num_1,num_2))
# import tensorflow as tf
# q = tf.sequence_mask([1,2,3,8], maxlen=5, dtype=tf.float32)
# with tf.Session() as sess:
#     print(sess.run(q))

# import jieba
# content = '违反交通运输管理法规，因而发生重大事故，致人重伤、死亡或者使公私财产遭受重大损失的，处三年以下有期徒刑或者拘役；交通运输肇事后逃逸或者有其他特别恶劣情节的，处三年以上七年以下有期徒刑；因逃逸致人死亡的，处七年以上有期徒刑。'
# p = jieba.lcut(content)
# print(p[-50:])
# print(len(p[-50:]))

import numpy as np
# array_1 = np.array([1,2,3,4])
# array_2 = np.array([1,2,3,4])
# array_3 = np.array([1,2,3,5])
# if (array_1 == array_2).all():
#     print("Yes")
# else:
#     print("No")
#
# print((array_1 == array_3))

# import tensorflow as tf
# array_1 = tf.constant([[[2,3],[4,5],[6,7]],[[2,5],[7,8],[1,10]]],dtype=tf.float32)
# max_num = tf.reduce_max(array_1,axis=-1,keep_dims=True)
# max_num = tf.keras.backend.repeat_elements(max_num,axis=-1,rep=2)
# # mean,var = tf.nn.moments(array_1,axes=-1)
# # mean_rep = tf.reshape(tf.keras.backend.repeat_elements(mean,axis=-1,rep=2),shape=[-1,3,2])
# # var = tf.reshape(tf.keras.backend.repeat_elements(var,axis=-1,rep=2),shape=[-1,3,2])
# with tf.Session() as sess:
#     print('mean',sess.run(max_num))


import json

# fr1 = open('resource/gyshz_traindata/word2vec/word_embedding.json','r',encoding='utf-8')
# fr2 = open('resource/gyshz_traindata/word2vec/vocab.txt','r',encoding='utf-8')
# word_dict = json.load(fr1)
# vocab = fr2.read().split('\n')
# output = []
# for v in vocab:
#     if v == '<UNK>':
#         continue
#     vector = ['0' for _ in range(128)]
#     if v in word_dict.keys():
#        vector = word_dict[v].split('\t')
#     output.append(' '.join([v] + vector))
#
# #
# fw = open('resource/gyshz_traindata/word2vec/vector_w2v.txt','w',encoding='utf-8')
# fw.write('\n'.join(output))









