import numpy as np
import matplotlib.pyplot as plt

 









vtilde_track = np.load('vtilde_track.npy')
v_track = np.load('v_track.npy')
gt_track = np.load('gt_track.npy')

i = 3
lenght = 1000

index = np.arange(lenght)
opacity = 1
bar_width = .2

plt.bar(index, gt_track[i,0,0, 0:lenght], bar_width, alpha=opacity,color='b', label='GT')

plt.bar(index + bar_width, v_track[i,0,0, 0:lenght], bar_width, alpha=opacity,color='r', label='V')

plt.bar(index + 2 * bar_width, vtilde_track[i,0,0, 0:lenght], bar_width, alpha=opacity,color='g', label='VTILDE')

plt.legend()