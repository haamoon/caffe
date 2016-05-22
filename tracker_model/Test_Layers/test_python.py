from __future__ import division
import scipy.sparse as ssp
import numpy as np
 
num_spatial_cells = 6


#set pooling_data, pooling_ind, pooling_ptr
N = 1
channels = 1
max_nrows = 6000
max_data_len = max_nrows
input_ncell = 200


nrows = np.random.random_integers(int(max_nrows/num_spatial_cells)) * num_spatial_cells
    
data_len_hi = np.min([max_data_len, nrows * input_ncell]) / 2
data_len_low = data_len_hi / 8
    
data_len = np.random.random_integers(data_len_low, data_len_hi)
rows = np.random.random_integers(0, nrows - 1, (data_len,))
cols = np.random.random_integers(0, input_ncell - 1, (data_len,))
data = np.random.rand(data_len)

pooling_matrix = ssp.csr_matrix((data, (rows, cols)), shape=(nrows, input_ncell), dtype='float64')
    
full_pooling_matrix = np.zeros((nrows, input_ncell), dtype='float64')
full_pooling_matrix[rows, cols] = data 
    
print np.abs(pooling_matrix.toarray() - full_pooling_matrix).max()

