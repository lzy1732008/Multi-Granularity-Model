from sklearn import metrics
import numpy as np
from processLaw import buildDataSetForRF
import json
from util import rules
import random
import re

fr = open('../resource/gyshz_lawDataSet.json', 'r', encoding='utf-8')
env = json.load(fr)
data = env['dataSet']
count = 0
qr = rules.QRulesEx()
hr = rules.HRulesEx()
pred_ys = []
target_ys = []
for sample in data:
    line, target_y = sample[0], int(sample[1])
    pattern = '，|。|；|：'
    regx = re.compile(pattern)
    array = regx.split(line)
    if len(array) == 2:
        input = array[0][1:]
    elif len(array) == 3:
        if line[0] == 'S':
            input = array[0][1:]
        else:
            input = array[1]
    else:
        input = array[1]

    if qr.predict3(input,line):
        y = 0
    elif hr.predict3(input,line):
        y = 1
    else:
        ran = random.Random()
        y = ran.randint(0,1)

    if y == target_y:
        count += 1

    target_ys.append(target_y)
    pred_ys.append(y)

print(metrics.classification_report(target_ys,pred_ys,digits=4))
print(metrics.accuracy_score(target_ys,pred_ys))





