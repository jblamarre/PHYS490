import numpy as np

def rand_points(n_points= 1, lb= -1, ub= 1):
    return np.random.uniform(lb, ub, n_points)

class Data():
    def __init__(self, u, v, lb, ub, dataset_size, look_back, epsilon):
        #Create random points to start sequences from
        random_x= rand_points(dataset_size)
        random_y= rand_points(dataset_size)
        
        #Normallize step
        epsilon = epsilon*(ub - lb)
        
        #Generate paths
        data_in, data_out = [], []
        for x, y in zip(random_x, random_y):
            points= [(x + epsilon*u(x, y), y + epsilon*v(x, y))]
            for i in range(look_back):
                #Make non linear evalutaion sequences
                theta = np.random.rand()*np.pi/2
                x1, y1= points[-1]
                points.append((x1 + epsilon*u(x1, y1)*np.cos(theta), 
                               y1 + epsilon*v(x1, y1)*np.sin(theta)))
            data_in.append(points[:-1])
            data_out.append(points[-1])
            
        self.data_in = np.array(data_in)
        self.data_out = np.array(data_out)
