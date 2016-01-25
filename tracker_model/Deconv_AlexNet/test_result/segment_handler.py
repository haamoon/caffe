import numpy as np
import scipy
from joblib import Parallel, delayed  
import multiprocessing

def get_segment_wrapper(segmentHandler, t, n, ind):
    return segmentHandler._get_segment(t,n, ind)

class SegmentHandler(object):
  
    def __init__(self, seg_data, sp_data, seg_ptr, sp_ptr, seg_num, sp_num, orig_size, image_size):
        self.seg_data = seg_data
        self.seg_ptr = seg_ptr
        self.seg_num = seg_num
    
        self.sp_data = sp_data
        self.sp_ptr = sp_ptr
        self.sp_num = sp_num
        
        self.orig_size = orig_size
        self.image_size = image_size
        self._create_segments()
        
    def get_colide_segments(self, t, n, r, c):
        if r > 0 and c > 0 and r < self.image_size[0] and c < self.image_size[1]:
            tmp_inds = np.where(self.segments[t][n][:, r, c] > 0)[0]
            segments = SegmentList(self.segments[t][n], tmp_inds)
            inds = self.inds[t][n][tmp_inds]
            return (segments, inds)
        return ([], [])

    def get_segment(self, t, n, ind):
        return self.segments[t][n][self.inds[t][n] == ind][0]
    
    def _create_segments(self):
        T = self.seg_data.shape[0]
        N = self.seg_data.shape[1]
        self.segments = np.empty((T,N,0)).tolist()
        self.inds = np.empty((T,N,0)).tolist()
        num_cores = multiprocessing.cpu_count()
        for t in range(T):
            for n in range(N):
                #for ind in range(self.seg_num[t,n]):
                #    print('processing segment t = %d n = %d ind = %d' % (t, n, ind) )
                #    segments[ind] = self._get_segment(t, n, ind)
                #(segmentHandler, t, n, ind)
                segments_list = Parallel(n_jobs=num_cores, verbose=0)(delayed(get_segment_wrapper)(self, t, n, i) for i in range(self.seg_num[t,n]))
                segments = np.stack(segments_list, axis = 0)
                
                ranks = segments.sum(1).sum(1)
                self.inds[t][n] = np.argsort(ranks)
                self.segments[t][n] = segments[self.inds[t][n], :, :]
            
    def _get_segment(self, t, n, ind):
        seg_data = self.seg_data[t,n]
        sp_data = self.sp_data[t, n]
        seg_start = int(self.seg_ptr[t,n,ind])
        seg_end = int(self.seg_ptr[t,n,ind+1])
        orig_size = self.orig_size[t,n]    
        all_rows = []
        all_cols = []
        segment = np.zeros((int(orig_size[0]), int(orig_size[1])))
        for i in range(seg_start, seg_end):
            sp_id = int(seg_data[i])
            sp_start = int(self.sp_ptr[t, n, sp_id])
            sp_end = int(self.sp_ptr[t, n, sp_id + 1])
            rows = sp_data[sp_start:sp_end,0]
            cols = sp_data[sp_start:sp_end,1]
            #rows = np.array(rows * (self.image_size[0] / orig_size[0]), 'int')
            #cols = np.array(cols * (self.image_size[1] / orig_size[1]), 'int')
            all_rows.extend(rows)
            all_cols.extend(cols)
        all_rows = [int(row) for row in all_rows]
        all_cols = [int(col) for col in all_cols]        
        segment[all_rows, all_cols] = 1
        return scipy.misc.imresize(segment, (int(self.image_size[0]), int(self.image_size[1])))
  
    def seg_num(self, t, n):
        return self.seg_num[t, n]

class SegmentList(object):
    def __init__(self, segments, selectedIndex):
        self.segments = segments
        self.inds = selectedIndex
    
    def __getitem__(self, index):
        return self.segments[self.inds[index]]
    
    def __len__(self):
        return len(self.inds)