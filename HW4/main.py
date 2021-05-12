# -*- coding: utf-8 -*-
# =============================================================================
# 
# PHYS490 - Winter 2021 - HW3 (J. Lamarre)
#
# =============================================================================

import json, argparse, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('src')
from torch.utils.data import DataLoader

#Converts +/- strings to +/- 1 resepectively
def true_data(in_file):

    #Fetch data
    f = open(in_file, "r")
    data_lines = f.readlines()
    f.close()

    #Initialize chain and chain length
    N = len(list(data_lines[0])) - 1
    chain = []

    #Use ascii code to convert +/- to 0/1
    for data_str in data_lines:
        for char in list(data_str)[:-1]:
            chain.append(44 - ord(char))

    #Reshape array
    chain = np.array(chain).reshape((-1,N))

    return chain, N

#Calculate energy of a chain
def chain_E(chain, J, axis = 1):
    return -np.sum(chain*np.roll(chain, -1, axis = 1)*J, axis = axis).reshape(-1,1)

#Metropolis-Hastings sampling algorithm
def MCMC(N, batch_size, J):

    #Ensure enough trials for proper sampling and generate random array of chains
    trials = N*N
    E_1 = (2*np.random.randint(0,2,size=(batch_size*N))-1).reshape(batch_size,N)

    #Compare and update E_1 array with random E_2 array with MH algorithm
    #to obatin lowest energy chains with some errors
    for t in range(trials):

        rnd = np.random.uniform(0,1)

        E_2 = np.reshape((2*np.random.randint(0,2,size=(batch_size*N))-1), (batch_size, N))
        E_1 = E_2*(chain_E(E_2, J) < chain_E(E_1, J))\
         + (E_2*(np.exp(chain_E(E_1, J) - chain_E(E_2, J)) >= rnd)\
          + E_1*(np.exp(chain_E(E_1, J) - chain_E(E_2, J)) < rnd))*(chain_E(E_2, J) >= chain_E(E_1, J))

    return E_1

#Traisn generative model trhough batch gradient descent
def train_model(data, epoch, N, batch_size, learning_rate, v, res_path):

    #Random initial coupler
    J = (2*np.random.randint(0,2,size=(N))-1).astype(float)
    KL = []

    #batch data
    data_batch = DataLoader(data, batch_size)

    #Run for all epochs
    for e in range(epoch):

        KL_batch = []

        #Run for all batches
        for ibatch, visible_batch in enumerate(data_batch):

            #True and sampled data
            data_D = visible_batch.detach().numpy()
            data_lbda = MCMC(N, batch_size, J)

            #Colum averages of x_i*x_j for both data sets
            D_avg = np.average(data_D*np.roll(data_D, -1, axis=1), axis = 0)
            lbda_avg = np.average(data_lbda*np.roll(data_lbda, -1, axis=1), axis = 0)

            #Update couplers
            J[:] = J[:] + learning_rate*(D_avg - lbda_avg)

            #Higher verbose KL calculation and tracking
            if v > 1:
                p = np.exp(-chain_E(data_D, J))/np.sum(np.exp(-chain_E(data_D, J, 1)))
                KL_batch.append(-1/batch_size * np.sum(np.log(p)))

        #Higher verbose KL calculation and tracking
        if v > 1:
            KL.append(np.average(KL_batch))

    #Higher verbose KL calculation and tracking
    if v > 1:
        plt.plot(range(len(KL)), KL)
        plt.xlabel('Epoch')
        plt.ylabel('KL divergence')
        plt.savefig(res_path + 'KL_div.png')

    return J

#Save couplers
def save_couplers(res_path, J):
    res_path = open(res_path + "results.json", "w")
    J = np.rint(J)
    json.dump({'(0, 1)': J[0], '(1, 2)': J[1], '(2, 3)': J[2], '(3, 0)': J[3]},
        res_path, sort_keys=True, indent=4)
    res_path.close()

#Initiate main
if __name__ == '__main__':
    
    parser= argparse.ArgumentParser(description='PHYS490 HW4 - Generative model.')
    
    
    parser.add_argument('--param', metavar='param.json',
                        help='file name for json attributes')
    parser.add_argument('--input-path', metavar='in.txt',
                    help='file name for input file of format +-...++')
    parser.add_argument('-v', type=int, default=1, metavar='N',
                        help='verbosity (default: 1)')
    parser.add_argument('--res-path', metavar='results',
                        help='path to save coupler results and KL divergence')
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)
        
    #Run generative model
    data, N = true_data(args.input_path)
    J = train_model(data, param['n_epoch'], N, 
        param['batch_size'], param['learning_rate'], args.v, args.res_path)
    save_couplers(args.res_path, J)



    
