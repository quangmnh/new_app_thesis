from matplotlib import pyplot as plt

import random as rd
a = []
for i in range(10):
    a.append(round(((24550-24370)*rd.random()+24370),11)/1554)
b = range(1,11)
plt.scatter(b,a, c="r")
plt.xlabel("Iterations")
plt.ylabel("Inference Time (ms)")
plt.legend(loc='upper left')
plt.ylim(min(a)*0.99,max(a)*1.01)
plt.show()