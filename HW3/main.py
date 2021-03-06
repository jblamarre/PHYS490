# -*- coding: utf-8 -*-
# =============================================================================
# 
# PHYS490 - Winter 2021 - HW3 (J. Lamarre)
#
# =============================================================================

import json, argparse, torch, sys
import torch.optim as optim
sys.path.append('src')
from nn import lstm_reg
from data import Data
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Generate data
def data_generation(u, v, lb, ub, data_set_size, seq_len, epsilon = 0.01):
    
    data = Data(u, v, lb, ub, data_set_size, seq_len, epsilon)
    
    return data

#Train model
def model_train(data, n_dim, seq_len, n_hidden, v, n_layers=1):
    
    #Formating data
    train_in = torch.from_numpy(data.data_in.reshape(-1, seq_len, 2))
    train_out = torch.from_numpy(data.data_out.reshape(-1, 1, 2))
    
    #Initialize model
    net = lstm_reg(n_dim, n_hidden, n_layers, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-2)
    
    #Train
    for e in range(2000):
        var_in = Variable(train_in).to(torch.float32)
        var_out = Variable(train_out).to(torch.float32)
        
        out = net(var_in)
        loss = criterion(out, var_out)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if v > 1:
          print(loss)
        elif (e + 1)% 100 == 0:
            print(loss)
            
    return net
    
#Test and compare model with ODE solver
def model_test(net, n_tests, u, v, lb, ub, res_path):
    
    #Create vector field
    net = net.eval ()
    x, y = np.meshgrid(np.linspace(lb, ub, 10), np.linspace(lb, ub, 10))
    plt.quiver(x, y, u(x, y), v(x, y))
    
    #Generate paths
    for j in range(n_tests):
        x = np.random.uniform(lb,ub)
        y = np.random.uniform(lb,ub)
        init_pt= torch.from_numpy(np.array((x, y)).reshape(1, 1, 2))
        init_pt= Variable(init_pt).to(torch.float32)
    
        model_pts= []
        for i in range(10000):
            init_pt= net(init_pt)
            model_pts.append(init_pt.detach().numpy())
           
        #Produce benchmark solution
        uv = lambda xy,t: [u(xy[0],xy[1]), v(xy[0],xy[1])]

        dt = (ub-lb)/100
        tf = (ub-lb)
        t = np.arange(0,tf+dt,dt)
        
        flow = odeint(uv,(x,y), t)
        color = np.random.rand(3,)
        model_pts= np.array(model_pts).reshape(-1, 2)
        
        #Plot solutions
        plt.plot(*np.array(model_pts).T, color = color)
        plt.plot(flow[:,0], flow[:,1], linestyle = '--',  color = color)
        plt.plot(x, y, 'o', markersize= 5,  color = color)
        
    plt.xlim([lb, ub])
    plt.ylim([lb, ub])
    
    if res_path != None:
        plt.savefig(res_path)

#Initiate main
if __name__ == '__main__':
    
    parser= argparse.ArgumentParser(description='PHYS490 HW3 - ODE solver.')
    
    
    parser.add_argument('--param', metavar='param.json',
                        help='file name for json attributes')
    parser.add_argument('-v', type=int, default=1, metavar='N',
                        help='verbosity (default: 1)')
    parser.add_argument('--res-path', metavar='results',
                        help='path to save the test plots at')
    parser.add_argument('--x-field', metavar='x**2',
                    help='expression of the x-component of the vector field')
    parser.add_argument('--y-field', metavar='y**2',
                    help='expression of the y-component of the vector field')
    parser.add_argument('--lb', default=-1, metavar='LB',
                        help='lower bound for initial conditions')
    parser.add_argument('--ub', default=1, metavar='UB',
                        help='upper bound for initial conditions')
    parser.add_argument('--n-tests', metavar='N_TESTS',
                        help='number of test trajectories to plot')
    args = parser.parse_args()
    
    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)
        
    lb = float(args.lb)
    ub = float(args.ub)
    
    #Functions
    u = lambda x, y: eval(args.x_field)
    v = lambda x, y: eval(args.y_field)
        
    #Run and tes RNN
    data = data_generation(u,v,lb,ub,param['ode']['data_set_size'],
                           param['ode']['seq_len'], param['ode']['epsilon'])
    net = model_train(data, param['model']['n_dim'],param['ode']['seq_len'],
                      param['model']['n_hidden'],args.v)
    model_test(net, int(args.n_tests), u, v, lb, ub, args.res_path)

    
