# import re
#
# # pattern = '，|。|；|：'
# # regx = re.compile(pattern)
# # input = "鼎折覆餗。asdhfjsdhfk；"
# # print(regx.split(input))
#
import tensorflow as tf
inputArray2 = tf.convert_to_tensor([[1.0,1.2,3.0],[2.1,2.2,3.2],[3.3,3.4,4.3],[0.3,0.4,1.4]])
inputArray1 = tf.convert_to_tensor([[9.0,10.2,11.0],[10.3,10.4,11.4],[11.5,11.6,12.6],[15.5,15.6,16.6]])
# splitArray = tf.convert_to_tensor([[0,1,2],[1,1,2]])
# gather_output = tf.gather_nd(inputArray,indices=splitArray[:,:2])
concat = tf.stack([inputArray1,inputArray2],axis=-1)

with tf.Session() as sess:
    t = sess.run(concat)
    print(t)


#统计0，1分类
# import json
# fr = open('resource/ft_labeled.json','r',encoding='utf-8')
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