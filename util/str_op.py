def showClassNum(content):
    lines = content.split('\n')
    class_0, class_1 = 0, 0

    for line in lines:
        if len(line.split('|')) <4:
            continue
        label = line.split('|')[3][0]
        if label == '0':
            class_0 += 1
        else:
            class_1 += 1
    print('0:',class_0)
    print('1:',class_1)
#
# f = open('../resource/val-init.txt','r',encoding='utf-8')
# lines = f.read()
# showClassNum(lines)
