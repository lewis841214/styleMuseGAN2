import numpy as np
from pypianoroll import Multitrack, Track
from multiprocessing import Pool, cpu_count
import queue
#from matplotlib import pyplot as plt
#import tensorflow as tf
import os
from multiprocessing import Process
import os
import time
i=0
k=0
c=20016
batch_size=500
#print (os.getcwd())
#print(os.getcwd())
#print('hisdfg')

rolls=[]
# analizing notes
# input a Multitrack object
# ex:
# another_multitrack = Multitrack('./data2/rock-39.mid')
# note_distribution(another_multitrack)
# return note_distribution of the whole song which order is C #C D #D ... bB B

def note_distribution(another_multitrack):
    # notes_per_beat is the minus 24 mod14 product C(Do) to be zero is better for the further task
    notes_per_beat = []
    length = len(another_multitrack.get_merged_pianoroll())
    merged = another_multitrack.get_merged_pianoroll()
    # print(merged)

    for i in range(length):
        # if (length-i)%1000==0:
        # print(length-i)
        # print('ori',np.nonzero(another_multitrack.get_merged_pianoroll()[i])[0])
        temp = list(np.nonzero(merged[i])[0])
        for j in range(len(temp)):
            temp[j] = (temp[j] - 24) % 12
        notes_per_beat.append(temp)
        # print(temp)
        # print('new',temp)
        # print(list(np.nonzero(another_multitrack.get_merged_pianoroll()[i])[0])+1)
    # np.nonzero(another_multitrack.get_merged_pianoroll())
    length = len(another_multitrack.get_merged_pianoroll())
    note_distributions = []
    for i in range(12):
        note_distributions.append(0)
    # print(note_distributions)
    for i in range(length):
        # print(notes_per_beat[i])
        for j in range(12):
            # print(j,notes_per_beat[i].count(j))
            note_distributions[j] = notes_per_beat[i].count(j) + note_distributions[j]
    # print(note_distributions)
    sum = 0
    for ele in note_distributions:
        sum = sum + ele
    note_distribution_frequency = []
    for ele in note_distributions:
        note_distribution_frequency.append(ele / sum)

    return note_distributions, note_distribution_frequency

############################
#note that 20016 is the smallest number 整除24 which's larger than 20000
######################################
def downsampling(number, roll):
    new = np.zeros((20016, 128))
    for c in range(int(20016 / number)):
        # 計算某個因出現的次數
        sdf = np.zeros(128)
        for j in range(128):
            for i in range(number):
                if roll[c * number + i][j] != 0:
                    sdf[j] = sdf[j] + roll[c * number + i][j]
                    # print('hi')
            # print(sdf)
        # 開始塞入new _roll which is a downsampling version of roll
        for j in range(128):
            for i in range(number):
                new[c * number + i][j] = sdf[j]
    return new


class LPDDataset:
    def __init__(self):
        with open('file_path2.txt', 'rb') as f:
            self.files = f.readlines()

    def __getitem__(self, idx):

        file = self.files[idx]
        # get original_dataset
        loaded = Multitrack(file[:-2].decode('utf-8'))
        roll = []
        for i in range(5):
            roll.append(loaded.tracks[i].pianoroll)
        # Then extend to 20016
        # 第一步 先偵測一下
        temproll = []
        """
         try:
            for i in range(5):
                print(roll[i].shape[0])
                temproll.append(np.vstack((roll[i], np.zeros((20000 - roll[i].shape[0], 128)))))

        except:
            print('length_greater_then_20016')
            return None
        """
        if roll[i].shape[0]>=30000:
            return None
        try:
            for i in range(5):
                #print(roll[i].shape[0])
                temproll.append(np.vstack((roll[i], np.zeros((30000 - roll[i].shape[0], 128)))))
        except:
            return None
        # temproll=np.vstack((data[i].tracks[2].pianoroll,temp))
        # temproll=np.vstack((temproll,np.zeros((30000-temproll.shape[0],128))))

        # Then do down sampling
        # 要一次把所有的down_sampling 都做完嗎?
        # 先做一個好了
        # lowest resolution
        #現在把所有的data 都轉成int16
        #但是在train的時候，需要轉回float
        for i in range(5):
            temproll[i] = downsampling(30000, temproll[i])
            temproll[i]=temproll[i].astype(np.int16)
       # rolls.append(temproll)
        return temproll

    def __len__(self):
        return len(self.files)




"""
def load(x):
    return x

#平行畫運算

from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor() as executer:
    for idx, output in enumerate(executer.map(load, data)):
        print(output)
        break


"""
datas = LPDDataset()
#single track np_array的call 法 第一個dimension為長度axis 第二個為音 axis
data_base=[]
print(len(datas))
def long_time_task(i):
    print('子进程: {} - 任务{}'.format(os.getpid(), i))
    time.sleep(2)
    print("结果: {}".format(8 ** 60))

#__name__='__test__'


def get_data(i):
    #print('子进程: {} - 任务{}'.format(os.getpid(), i))
    return datas[i]
import pickle

if __name__=='__test__':
    print(cpu_count())

if __name__=='__main__':

    """
    q = queue.Queue()
    print("CPU内核数:{}".format(cpu_count()))
    print('当前母进程: {}'.format(os.getpid()))
    start = time.time()
    p = Pool(cpu_count())
    for i in range(2):
        q.put(p.apply_async(get_data, args=(i,)))
        # print('当前子进程: {}'.format(os.getpid()))
    print('等待所有子进程完成。')
    p.close()
    p.join()
    end = time.time()
    print("总共用时{}秒".format((end - start)))

    while q.empty() == False:
        rolls.append(q.get().get())
 """
    #print(rolls)

    q=queue.Queue()
    #print("CPU内核数:{}".format(cpu_count()))
    #print('当前母进程: {}'.format(os.getpid()))
    start = time.time()

    for j in range(int(21425/7)):
        j=j+2144
        print('{} start'.format(j))
        p = Pool(cpu_count() - 1)
        for i in range(7):
            q.put(p.apply_async(get_data, args=(i+10*j,)))
            #print('当前子进程: {}'.format(os.getpid()))

        while q.empty() == False:
            rolls.append(q.get().get())
        #print('hi')
        #print(rolls)
        print('{} dump'.format(j))

        #D:\1projects\MusicGan\styleMuseGAN\down_sampling_data
        #CHange to j-1
        pickle.dump(rolls, open('D:/1projects/MusicGan/styleMuseGAN/down_sampling_data/d_30000_{}.pke'.format(j-1), 'wb'))
        rolls=[]
        p.close()
        p.join()
    #print('等待所有子进程完成。')

        end = time.time()
        print("总共用时{}秒".format((end - start)))

    ############################################################
    ############################################################
    #arr.astype(int16)
    #要使用時再轉回(float64)
    # 現在把所有的data 都轉成int16
    # 但是在train的時候，需要轉回float
    ############################################################
    ############################################################

    """
    print('当前母进程: {}'.format(os.getpid()))
    start = time.time()
    print(start)
    for i in range(2):
        long_time_task(i)

    end = time.time()
    print(end)

    print("用时{}秒".format((end                                   - start)))
    """



    ###
    """
     print('当前母进程: {}'.format(os.getpid()))
    start = time.time()
    p1 = Process(target=long_time_task, args=(1,))
    p2 = Process(target=long_time_task, args=(2,))
    print('等待所有子进程完成。')
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    end = time.time()
    print("总共用时{}秒".format((end - start)))
    """




"""
    i=0
    print('当前母进程: {}'.format(os.getpid()))
    start = time.time()
    for d in datas:

        if i==2:
            break;
        print(i)
        data_base.append(d)
        i = i + 1
    end=time.time()
    print("总共用时{}秒".format((end - start)))
    print('###')
    print('当前母进程: {}'.format(os.getpid()))
    start = time.time()
    p1 = Process(target=get_data, args=(1,))
    p2 = Process(target=get_data, args=(2,))
    print('等待所有子进程完成。')
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    end = time.time()
    print("总共用时{}秒".format((end - start)))
"""



