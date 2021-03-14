# =============================================================================
#
# PHYS490 - Winter 2021 - HW2 (J. Lamarre) 
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):
    '''
    Neural network class.
    Architecture:
        Three fully-connected layers fc1, fc2 and fc3.
        Three nonlinear activation functions relu and softmax.
    '''
    def __init__(self, n_bits):
        super(Net, self).__init__()
        self.fc1= nn.Linear(n_bits**2, 256)
        self.fc2= nn.Linear(256,64)
        self.fc3= nn.Linear(64,5)
    
    # Feedforward function
    def forward(self, x):
        h= func.relu(self.fc1(x))
        z= func.relu(self.fc2(h))
        y= func.softmax(self.fc3(z),dim=1)
        return y
    
    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
       
    # Backpropagation function
    def backprop(self, data, loss, epoch, optimizer):
        self.train()
        inputs= torch.from_numpy(data.x_train)
        targets= torch.from_numpy(data.y_train)
        outputs= self(inputs)
        obj_val= loss(self.forward(inputs), targets)
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()
    
    # Test function. Avoids calculation of gradients.
    def test(self, data, loss, epoch):
        self.eval()
        with torch.no_grad():
            inputs= torch.from_numpy(data.x_test)
            targets= torch.from_numpy(data.y_test)
            outputs= self(inputs)
            cross_val= loss(self.forward(inputs), targets)
        return cross_val.item()
