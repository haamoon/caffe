import numpy as np
from mouse_handler import MouseHandler
from segment_handler import SegmentHandler

blobs = net.blobs
image = blobs['image'].data.copy().swapaxes(1,2).swapaxes(2,3)
mean_value = np.array([104.00699, 116.66877, 122.67892]).reshape(1,1,1,3)
image += mean_value
image /= 255.0
if False:
    net.forward()

t = 0
n = 0
N = blobs['v'].data.shape[1]

segment_handler = SegmentHandler(blobs['seg_data'].data, 
                                 blobs['spixel_data'].data,
                                    blobs['seg_ptr'].data,
                                    blobs['spixel_ptr'].data,
                                    blobs['seg_num'].data,
                                    blobs['spixel_num'].data,
                                    blobs['mask_size'].data,
                                    image.shape[1:3])


mouse_handler = MouseHandler(image, blobs['v'].data, 
                             blobs['vtilde'].data, segment_handler,0, 0, N)
