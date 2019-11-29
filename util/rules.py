#该文件是为了通过正则化方式抽取特征
import re
class Rules:
    def __init__(self,x):
        self.x = x

    #以某些词开头的
    def rule1(self, words):
        for w in words:
            if str(self.x).startswith(w):
                return True
        return False

    #以某些词结尾
    def rule2(self,words):
        for w in words:
            if str(self.x).endswith(w):
                return True
        return False

    #"XXX"这种名词性规则
    def rule3(self):
        pattern = r"^\"\S+\"$"
        if re.search(pattern,self.x, re.M|re.I):
            return True
        return False

    #包含某些词的
    def rule4(self,words):
        for w in words:
            if str(self.x).count(w) > 0:
                return True
        return False



class QRules:
    #以某些词开头
    def rule1(self,input):
        words = ["如果","根据","对于","是指"]
        rule = Rules(input)
        return rule.rule1(words)

    #以某些词结尾
    def rule2(self,input):
        words = ["除外","时","的"]
        rule = Rules(input)
        return rule.rule2(words)

    #包含某些词
    def rule3(self,input):
        words = ["下列情形"]
        rule = Rules(input)
        return rule.rule4(words)

    def inter(self,input):
        return [int(self.rule1(input)),int(self.rule2(input)),int(self.rule3(input))]

class HRules:
    # 以某些词开头
    def rule1(self, input):
        words = ["应该", "可以", "也可以","不得"]
        rule = Rules(input)
        return rule.rule1(words)

    # 包含某些词
    def rule2(self, input):
        words = ["是指","应当按规定"]
        rule = Rules(input)
        return rule.rule4(words)

    # 名词解释
    def rule3(self,input):
        rule = Rules(input)
        return rule.rule3()

    def inter(self,input):
        return [int(self.rule1(input)),int(self.rule2(input)),int(self.rule3(input))]



