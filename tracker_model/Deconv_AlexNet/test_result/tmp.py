import numpy as np
import matplotlib.pyplot as plt



i = 3
lenght = 1000 
gt_num = 2

v = net.blobs['v'].data
vtilde = net.blobs['vtilde'].data
gt_overlaps = net.blobs['gt_overlaps'].data





track_ids = np.argmax(gt_overlaps[0], 2)[0, :gt_num]

vtilde_track = vtilde[:, :, track_ids, :]
v_track = v[:, :, track_ids, :]
gt_track = gt_overlaps[:, :, :gt_num, :]



index = np.arange(lenght)
opacity = 1
bar_width = .2

plt.bar(index, gt_track[i,0,0, 0:lenght], bar_width, alpha=opacity,color='b', label='GT')

plt.bar(index + bar_width, v_track[i,0,0, 0:lenght], bar_width, alpha=opacity,color='r', label='V')

plt.bar(index + 2 * bar_width, vtilde_track[i,0,0, 0:lenght], bar_width, alpha=opacity,color='g', label='VTILDE')

plt.legend()
