import sys
import os
import json
import numpy as np

#Analytic solution for optimal w
def w_analityc(x,T,n,m):
    
    #Psi matrix, n x m
    PSI = np.ones([n,m])
    PSI[:,1:] = x
    
    #Target solutions
    T = np.array(T).reshape((n,1))
    
    #(Psi^T * Psi)^-1 * Psi^T * T - Analytical solution
    return np.matmul(np.matmul(np.linalg.inv(
        np.matmul(np.transpose(PSI),PSI)),np.transpose(PSI)),T)

#Gradient descent solutionn for optimal w
def w_gd(x,T,n,m,alpha,niter):
    
    #Initial paramter guess
    w_gd0 = np.ones(m)
    
    #Psi matrix - used to simplify code
    PSI = np.ones([n,m])
    PSI[:,1:] = x

    #Batch gradient descent with provided hyperparameters
    for i in range(niter):
        w_gd0 = w_gd0 - alpha*np.sum(
            -(T - np.sum(w_gd0*PSI, axis = 1)).reshape(n,1)*PSI, axis = 0)
    return w_gd0

#Main function
def main():
    #Files provided at input
    filepath_in = sys.argv[1]
    filepath_json = sys.argv[2]
    
    #Parameter initialisation
    x = []  #Samples
    T = []  #Target values
    n = 0   #Number of samples
    m = 0   #Number of features
    
    #Retreive data, target values, m - number of features, n - data set length
    f = open(filepath_in, "r")
    for line in f:
        x_raw = line.split()
        for i in x_raw[:-1]:
            x.append(float(i))
        T.append(float(x_raw[-1]))
        m = len(x_raw)
        n+=1
    f.close()
    
    #Formating data
    x = np.array(x).reshape(n,m-1)
    
    #Retreive hyperparameters
    with open(filepath_json) as f:
        hyperparam = json.load(f)
    
    #Find solutions
    w_anal_sol = w_analityc(x,T,n,m)
    w_gd_sol = w_gd(x,T,n,m,hyperparam["learning rate"],hyperparam["num iter"])
    
    #Output optmized parameters as input_filename.out
    filename = os.path.splitext(filepath_in)[0]
    f = open(filename+ ".out", "w+")
    for i in w_anal_sol:
        f.write(str(i[0]) + '\n')
    f.write('\n')
    for i in w_gd_sol:
        f.write(str(i) + '\n')
    f.close()

#Initiate main
if __name__ == '__main__':
    main()
