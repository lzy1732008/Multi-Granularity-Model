from multiprocessing import Process,Queue,Pool


import time
startTime = time.time()

q = Queue()
def count(start,end):
    q.put(range(start,end))


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
            p.join()


mp = MultiProcess(tar=[count,count,count],arg=[(0,10),(10,20),(20,30)])
mp.multi_processing()
result = []
while not q.empty():
    result += list(q.get())


# p = Pool(5)
# for i in range(5):
#     p.apply_async(count, args=(i*5,i*5 + 5))
#     print('Waiting for all subprocesses done...')
#     p.close()
#     p.join()
# print(list(q.get()))

# from multiprocessing import Pool
# import os, time, random
#
# def long_time_task(name):
#     print('Run task %s (%s)...' % (name, os.getpid()))
#     start = time.time()
#     time.sleep(random.random() * 3)
#     end = time.time()
#     print('Task %s runs %0.2f seconds.' % (name, (end - start)))
#
# if __name__=='__main__':
#     print('Parent process %s.' % os.getpid())
#     p = Pool(5)
#     for i in range(5):
#         p.apply_async(count, args=(i* 5,i*5 +5))
#     print('Waiting for all subprocesses done...')
#     p.close()
#     p.join()
#     print('All subprocesses done.')
#     while not q.empty():
#         print(list(q.get()))