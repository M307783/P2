# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:31:20 2026

@author: Ludvig
"""

import math
import matplotlib.pyplot as plt
import numpy as np




img = np.array([[math.sqrt(13), math.sqrt(85), math.sqrt(17)], [math.sqrt(13), math.sqrt(89), math.sqrt(26)]])

plt.imshow(img, cmap='gray')
plt.yticks([])
plt.xticks([])
plt.title('Grayscale Representation of k-space Matrix')
plt.savefig('Grayscale Representation of k-space Matrix.png', transparent=True)
'''
scale = np.linspace(0,10, 11)
scale2 = np.vstack((scale, scale))
billede = plt.imshow(scale2, cmap='gray',rasterized=True)
plt.axis([0,10,0,1])
plt.yticks([])
plt.title('Grayscale')
plt.savefig('Grayscale.png', transparent=True)
'''
