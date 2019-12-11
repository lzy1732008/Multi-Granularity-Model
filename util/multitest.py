from multiprocessing import Process,Queue

import time
startTime = time.time()



class MultiProcess(object):
    target = []
    args = []

    def __init__(self, tar, arg):
        self.target = tar
        self.args = arg

    def multi_processing(self):
        jobs = []
        process_num = len(self.target)
        for i in range(process_num):
            p = Process(target=self.target[i], args=self.args[i])
            jobs.append(p)
            p.start()


