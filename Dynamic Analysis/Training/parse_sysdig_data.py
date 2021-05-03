from datetime import datetime,timedelta

syscalls = dict()
with open('syscall.csv') as f:
    for line in f:
        n,s = line.split(',')
        n = int(n)
        s = s.strip()
        syscalls[s] = n

current_time = None
current_count = 1
with open('attack_sysdig.txt') as f:
    for line in f:
        t,s = line.split(' ')
        s = s.strip()
        if s not in syscalls:
            continue
        t = t[:-3]
        dt = datetime.strptime(t,"%H:%M:%S.%f")
        sn = syscalls[s]
        if current_time is None:
            current_time = dt
            ft = open('./Dataset/Attack'+str(current_count)+'.txt','a+')
        if dt - current_time < timedelta(milliseconds=100):
                ft.write(str(sn)+" ")
        else:
            ft.close()
            current_count += 1
            current_time = dt
            ft = open('./Dataset/Attack'+str(current_count)+'.txt','a+')
