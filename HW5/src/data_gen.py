# =============================================================================
# PHYS490 - Winter 2021 - HW5 (J. Lamarre) 
# =============================================================================

#-------------------------------Imports----------------------------------------

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

#----------------------------Data batching-------------------------------------

def batched_data(n_samples, filepath_data, batch_size, kwargs):
    
    #Fetch data
    f = open(filepath_data, "r")
    data_str = f.readlines()

    #Generate randome sample selection
    n_lines = len(data_str)
    test_samples = np.random.randint(0,n_lines, n_samples)

    #Prepare blank target and 14 x 14 image arrays
    x_train= np.zeros((n_lines,196), dtype= np.float32)
    y_train= np.zeros((n_lines,5), dtype= np.float32)
    
    #Assign data to respective array
    for c in range(n_lines):
        data_temp = [int(s) for s in data_str[c].split(' ')]
        y_train[c,int(data_temp.pop(-1)/2)] = 1
        x_train[c,:] = data_temp
    f.close()

    #Normalize data
    x_train= x_train/255
    x_test = x_train[test_samples, :]
    y_test = y_train[test_samples, :]

    #Batch data into container
    data_train = DataLoader(TensorDataset(Tensor(x_train),Tensor(y_train)),
                            batch_size = batch_size, **kwargs)
    data_test =  DataLoader(TensorDataset(Tensor(x_test),Tensor(y_test)), **kwargs)
    
    return data_train,  data_test
