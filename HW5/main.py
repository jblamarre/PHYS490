import json, argparse, torch, sys, os
import torch.optim as optim
import matplotlib.pyplot as plt
sys.path.append('src')
from nn_gen import VAE
from data_gen import batched_data


def loss_func(criterion, re_x, x, mu, logsig):

    KL = -0.5*torch.sum(1 + logsig - mu.pow(2) - logsig.exp())
    BCE = criterion(re_x, x.view(-1, 196))
    
    return KL + BCE


def train(model, optimizer, criterion, data, device, epoch, verbosity):

    training_loss = 0

    model.train()

    #Run for all batches
    for ibatch, (data, _) in enumerate(data):

        data = data.to(device)
        optimizer.zero_grad()
        re_xbatch, mu, logsig = model(data)
        loss = loss_func(criterion, re_xbatch, data, mu, logsig)
        loss.backward()
        training_loss += loss.item()
        optimizer.step()

    if verbosity > 1:
        print("Epoch: {}, Training Loss: {:.4f}".format(epoch, training_loss))

    return training_loss

def test(model, optimizer, criterion, data, device, res_path):
    
    test_loss = 0

    model.eval()

    with torch.no_grad():
        
        for ibatch, (data, _) in enumerate(data):

            data = data.to(device)
            re_xbatch, mu, logsig = model(data)
            test_loss += loss_func(criterion, re_xbatch, data, mu, logsig).item()

            fig = plt.figure()
            plt.imshow(re_xbatch.view(14, 14).cpu(), cmap='gray')
            fig.savefig(res_path + '/' +str(ibatch) + '.pdf')

#Initiate main
if __name__ == '__main__':
    
    parser= argparse.ArgumentParser(description=
                                'PHYS490 HW2 - Digit recognition of even \
                                    numbers using MNIST dataset and pytorch.')
                                
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

    if not os.path.exists(args.res_path):
        os.mkdir(args.res_path)

    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    model = VAE().to(device)

    # Define an optimizer and the loss function
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])
    criterion = torch.nn.BCELoss(reduction= 'sum')

    data_train, data_test = batched_data(args.n, param['filepath_data'],
                                         int(param['batch_size']), kwargs)

    epochs = param['num_epochs']
    verbosity = args.v

    training_loss = []

    for e in range(epochs):

        training_loss.append(train(model, optimizer, criterion, data_train, device, e, verbosity))
    
    if args.n > 0:
        test(model, optimizer, criterion, data_test, device, args.res_path)

    fig = plt.figure()
    plt.plot(range(epochs), training_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    fig.savefig(args.res_path + '/loss' + '.pdf')



