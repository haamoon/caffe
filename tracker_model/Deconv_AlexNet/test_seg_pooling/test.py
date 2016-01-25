import numpy as np
import scipy as sp


blobs = solver.net.blobs;
#solver.net.forward()

seg_feature = blobs['seg_feature'].data[0, 0]
conv_feature = blobs['conv-feature'].data[0]

mask_num = 100
##############################start

seg_data = blobs['seg_data'].data[0,0]
sp_data = blobs['spixel_data'].data[0, 0]
seg_start = blobs['seg_ptr'].data[0,0,mask_num]
seg_end = blobs['seg_ptr'].data[0,0,mask_num+1]
mask_size = blobs['mask_size'].data[0,0]
seg_coef = blobs['seg_coef'].data[0, 0]
avg = 0

all_rows = []
all_cols = []
#tmp = np.zeros((3, blobs['image'].data.shape[2], blobs['image'].data.shape[3]))
for i in range(seg_start, seg_end):
    sp_id = int(seg_data[i])
    sp_start = blobs['spixel_ptr'].data[0, 0, sp_id]
    sp_end = blobs['spixel_ptr'].data[0, 0, sp_id + 1]    
    rows = sp_data[sp_start:sp_end,0]
    cols = sp_data[sp_start:sp_end,1]
    
    rows = np.array(rows * (blobs['image'].data.shape[2] / mask_size[0]), 'int')
    cols = np.array(cols * (blobs['image'].data.shape[3] / mask_size[1]), 'int')
    all_rows.extend(rows)
    all_cols.extend(cols)
    
    #tmp[0, rows, cols] = np.random.uniform(0, 255)
    #tmp[1, rows, cols] = np.random.uniform(0, 255)
    #tmp[2, rows, cols] = np.random.uniform(0, 255)
    
    avg += seg_coef[i] * conv_feature[:, rows, cols].sum(1) / len(rows)

#sp.misc.toimage(tmp).show()

indirect_rows = np.array(all_rows, 'int')
indirect_cols = np.array(all_cols, 'int') 
indirect_mask = np.zeros((blobs['image'].data.shape[2], blobs['image'].data.shape[3]))
indirect_mask[indirect_rows, indirect_cols] = 1
sp.misc.toimage(indirect_mask).show()

indirect_sum = (avg - seg_feature[mask_num, :]).sum()
print indirect_sum
###############################end

orig_mask = np.load('mask_orig.npy').transpose([2, 0, 1])
size = orig_mask.sum(1).sum(1)
if (size == 0).any():
    orig_mask = orig_mask[size > 0]
mask_elem = np.where(orig_mask[mask_num] == 1)

direct_rows = np.array(mask_elem[0] * (blobs['image'].data.shape[2] / float(orig_mask.shape[1])), 'int')
direct_cols = np.array(mask_elem[1] * (blobs['image'].data.shape[3] / float(orig_mask.shape[2])), 'int')

phy_seg_feature = conv_feature[:, direct_rows, direct_cols].sum(1) / len(direct_rows)

#direct_mask = np.zeros((blobs['image'].data.shape[2], blobs['image'].data.shape[3]))
#direct_mask[direct_rows, direct_cols] = 1
#sp.misc.toimage(direct_mask).show()

direct_sum = (phy_seg_feature - seg_feature[mask_num, :]).sum()



#phy_seg_feature = conv_feature[0,  :, rows, cols].sum(0) / len(rows)
#caf_seg_feature = seg_feature[0,0,mask_num, :]
