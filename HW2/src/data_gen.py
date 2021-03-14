# =============================================================================
#
# PHYS490 - Winter 2021 - HW2 (J. Lamarre) 
#
# =============================================================================
import numpy as np

class Data():
    def __init__(self,n_bits, n_training_data, n_test_data, filepath_data):
        
        #Prepare blank target and 14 x 14 image arrays
        x_temp= np.zeros((n_training_data+n_test_data,n_bits**2),dtype= np.float32)
        y_temp= np.zeros((n_training_data+n_test_data,5),dtype= np.float32)
        
        #Fetch data
        f = open(filepath_data, "r")
        data_str = f.readlines()
        
        #Make sure user defined parameters for batch size are not too big
        assert n_training_data+n_test_data>=np.size(data_str), \
            'Batch size bigger then size of data set'
        
        #Assign data to respective array
        for c in range(n_training_data+n_test_data):
            data_temp = [int(s) for s in data_str[c].split(' ')]
            y_temp[c,int(data_temp.pop(-1)/2)] = 1
            x_temp[c,:] = data_temp
        f.close()
        
        #Assign to instance - normalize greyscale
        self.x_train= x_temp[0:n_training_data,:]/255
        self.y_train= y_temp[0:n_training_data,:]
        self.x_test= x_temp[n_training_data:,:n_training_data+n_test_data]/255
        self.y_test= y_temp[n_training_data:,:n_training_data+n_test_data]
