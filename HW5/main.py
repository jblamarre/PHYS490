# =============================================================================
# PHYS490 - Winter 2021 - HW5 (J. Lamarre) 
# =============================================================================

#-------------------------------Imports----------------------------------------

import json, argparse, torch, sys, os
import torch.optim as optim
import matplotlib.pyplot as plt
sys.path.append('src')
from nn_gen import VAE
from data_gen import batched_data

#----------------------------Loss function-------------------------------------

def loss_func(criterion, re_x, x, mu, logsig):

    #KL divergence and BCE loss to minimize
    KL = -0.5*torch.sum(1 + logsig - mu.pow(2) - logsig.exp())
    BCE = criterion(re_x, x.view(-1, 196))
    
    return KL + BCE

#------------------------------Training----------------------------------------

def train(model, optimizer, criterion, data, device, epoch, verbosity):

	#Loss initialisation and training mode
    training_loss = 0

    model.train()

    #Iterate over all bacthes
    for ibatch, (data, _) in enumerate(data):

        #Train model
        data = data.to(device)
        optimizer.zero_grad()
        re_xbatch, mu, logsig = model(data)
        loss = loss_func(criterion, re_xbatch, data, mu, logsig)
        loss.backward()
        training_loss += loss.item()
        optimizer.step()

    #Loss tracking
    if verbosity > 1:
        print("Epoch: {}, Training Loss: {:.4f}".format(epoch, training_loss))

    return training_loss

#--------------------------Sample generation-----------------------------------

def test(model, optimizer, criterion, data, device, res_path):

    #Turn off training
    model.eval()

    with torch.no_grad():
        
        #Iterate across randomly selected samples
        for ibatch, (data, _) in enumerate(data):

            #Sample generation from data
            data = data.to(device)
            re_xbatch, mu, logsig = model(data)

            #Image reconstruction and saving to results file
            fig = plt.figure()
            plt.imshow(re_xbatch.view(14, 14).cpu(), cmap='gray')
            fig.savefig(res_path + '/' +str(ibatch) + '.pdf')

#----------------------------Main function-------------------------------------

if __name__ == '__main__':
    
    #Argument parser
    parser= argparse.ArgumentParser(description=
                                'PHYS490 HW5 - Digit generation of even \
                                    numbers using MNIST dataset and VAE.')
                                
    parser.add_argument('--param', metavar='param.json',
                        help='parameter file name')
    parser.add_argument('-v', type=int, default=1, metavar='N',
                        help='verbosity (default: 1)')
    parser.add_argument('--res-path', metavar='results',
                        help='path of results')
    parser.add_argument('-n',type=int, default = 10, metavar='100',
                        help='number of samples (default: 10')
    parser.add_argument('--cuda', type=bool, default=False, metavar='True',
                        help='GPU usage on CUDA (default: False)')
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)

    #Create results directory if it dosn't already exist
    if not os.path.exists(args.res_path):
        os.mkdir(args.res_path)

    #Device selection (CPU or NVIDIA GPU)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    #VAE initialisation
    model = VAE().to(device)

    # Define an optimizer and the criterion
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])
    criterion = torch.nn.BCELoss(reduction= 'sum')

    #Batching of training data and radom selection for sampling data
    data_train, data_test = batched_data(args.n, param['filepath_data'],
                                         int(param['batch_size']), kwargs)

    #Epochs and verbosity
    epochs = param['num_epochs']
    verbosity = args.v

    training_loss = []

    #Training and loss tracking
    for e in range(epochs):

        training_loss.append(train(model, optimizer, criterion, data_train, device, e, verbosity))
    
    #Sample generation
    if args.n > 0:
        test(model, optimizer, criterion, data_test, device, args.res_path)

    #Training loss ploting
    fig = plt.figure()
    plt.plot(range(epochs), training_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    fig.savefig(args.res_path + '/loss' + '.pdf')



