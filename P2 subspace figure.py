import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return x*2

x = np.linspace(-2,2,4)
yaxis = [np.min(func(x)),np.max(func(x))]

plt.plot(x,func(x), 'blue')
plt.plot(np.zeros(len(yaxis)),yaxis,'black')
plt.plot(x,np.zeros(len(x)),'black')
plt.plot(0,0,marker='o',color='green')
plt.grid(True)
plt.xticks(np.linspace(-2,2,5))
plt.yticks(np.linspace(-4,4,5))
plt.xlim(-2, 2), plt.ylim(-4, 4)
plt.text(0.32,1.05,'H',color='blue',weight='bold')
plt.text(0.05,-0.36,'0',color='green',weight='bold')
plt.savefig('Subspace figur.png',transparent=True)