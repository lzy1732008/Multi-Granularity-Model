#1、 展示词性标注的效果
#2、 展示经过分词、过滤后的效果

import jieba.posseg as pos
from zhon.hanzi import punctuation
import jieba

def preprocessText(line):
    # output = pos.lcut(line)
    # output_cx = []
    # for w,cx in output:
    #     output_cx.append('{0}:{1}'.format(w,cx))
    # print(' '.join(output_cx))

    # output_str = []
    # for w, cx in output:
    #     if cx in ['n','a','v']:
    #         output_str.append(w)
    # print(' '.join(output_str))

    output = jieba.lcut(line)
    print(' '.join(output))


# line = '公诉机关指控，2013年7月6日20时，被告人曹某驾驶小型客车沿漳河旅游公路由东向西行驶，至周河村五组路段时，与前方叶某（男，殁年61岁）驾驶的无号牌手持拖拉机相撞，造成叶某受伤后抢救无效死亡的交通事故'
# line = '拘役的缓刑考验期限为原判刑期以上一年以下，但是不能少于二个月。有期徒刑的缓刑考验期限为原判刑期以上五年以下，但是不能少于一年。缓刑考验期限，从判决确定之日起计算。'
# preprocessText(line)

lines = list(map(lambda x: x.strip(), open('../resource/stopwords.txt', 'r', encoding='utf-8').read().split('\n')))
index = 0
while index < len(lines):
    end_index = index + 20
    end_index = min(end_index, len(lines))
    print('  '.join(lines[index:end_index]))
    index += 20
