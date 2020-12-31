import matplotlib.pyplot as plt
import numpy as np

with open('log.txt','r') as f:
    list = f.readlines()
    f.close()

cost_his = []
for item in list:
    item = item.split('\n')[0]
    loss = item.split(',')[0]
    loss = float(loss[1:len(loss)])
    acc = item.split(',')[1]
    acc = float(acc[0:len(acc)-1])
    cost_his.append([loss])

plt.plot(np.arange(len(cost_his)),cost_his)
plt.ylabel('Cost')
plt.xlabel('training steps')
plt.show()