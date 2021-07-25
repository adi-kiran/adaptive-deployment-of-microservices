
import pandas as pd
import os
import sys
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from subprocess import Popen, PIPE
from datetime import datetime,timedelta
from collections import defaultdict
import subprocess
import pickle

filename = 'finalized_model_pers.sav'
 
# columns = [ 0,  1,  2,  3,  4,  5,  7,  8,
#    9, 10, 11, 12, 13, 14, 16, 21,
#   22, 32, 39, 41, 42, 43, 44, 45,
#   47, 51, 54, 55, 56, 59, 61, 79,
#   89,102,104,105,107,108,110,125,
#  126,157,158,186,202,217,257,269,
#  273, 38, 63, 72, 83, 87, 91,
#   95, 99,111,137,218,221,230,231,
#  268, 28, 48, 60, 24, 23]

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

syscalls = dict()
with open('syscall.csv') as f:
    for line in f:
        n,s = line.split(',')
        n = int(n)
        s = s.strip()
        syscalls[s] = n

columns = [j for i,j in syscalls.items()]
# your_command = ['docker', 'exec', 'sysdig', 'sysdig', '-p', '%evt.time %evt.type', 'container.id=fddf78cab5fa']
#your_command = ['ping', '8.8.8.8']
current_time = None
# p = subprocess.Popen(your_command, stdout=subprocess.PIPE)
f = open("test.txt","r")
l = f.readlines()
f.close()
freq = defaultdict(int)
for line in l:
    # print('>>> {}'.format(line.decode('utf-8').rstrip()))
    # print(line.decode('utf-8').rstrip().split(' '))
    # t,s = line.decode('utf-8').strip('"|\n').split(' ')
    t,s = line.strip('"|\n').split(' ')
    s = s.strip()
    # print(t,s)
    if s not in syscalls:
        continue
    t = t[:-3]
    dt = datetime.strptime(t,"%H:%M:%S.%f")
    sn = syscalls[s]
    if current_time is None:
        current_time = dt
    if dt - current_time < timedelta(milliseconds=100):
        freq[sn] += 1
    else:
        df_res = pd.DataFrame(columns=columns)
        df_res =  df_res.append(freq, ignore_index=True)
        df_res = df_res.fillna(0)
        res = loaded_model.predict(df_res)
        # print(res)
        if res[0] == 1:
            print("ATTACK detected")
        # else:
        #     print("No attack")
        current_time = dt
        freq.clear()
print("done")