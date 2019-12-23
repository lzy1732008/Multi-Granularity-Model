#把与刑法133条匹配的，label为1的随机删除掉一半
import models.parameter as param
import random
def rmLabel1InLaw():
    ft = '中华人民共和国刑法(2015)第一百三十三条:违反交通运输管理法规，因而发生重大事故，致人重伤、死亡或者使公私财产遭受重大损失的，处三年以下有期徒刑或者拘役；交通运输肇事后逃逸或者有其他特别恶劣情节的，处三年以上七年以下有期徒刑；因逃逸致人死亡的，处七年以上有期徒刑。'
    lines = open('../resource/val-init.txt','r',encoding='utf-8').read().split('\n')
    newlines = []
    for line in lines:
        line = line.strip()
        if line != "":
            items = line.split('|')
            if items[2] == ft and items[-1] == '1':
                pw = random.random()
                if pw > 0.5:
                #save this sample
                   newlines.append(line)
            else:
                newlines.append(line)


    open('../resource/val-删除1.txt','w',encoding='utf-8').write('\n'.join(newlines))

rmLabel1InLaw()
