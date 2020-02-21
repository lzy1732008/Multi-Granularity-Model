import models.parameter as param
def getwslist():
    lines = open('../../resource/gyshz_traindata/test-init.txt','r',encoding='utf-8').read().split('\n')
    namels = []
    for i in range(len(lines)):
        line = lines[i]
        if line.strip() == "":
            continue
        array = line.split('|')

        if len(array) < 4:
            continue

        namels.append(array[0])
    return namels