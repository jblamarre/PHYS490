# =============================================================================
# PHYS490 - Winter 2021 - HW5 (J. Lamarre) 
# =============================================================================

#-------------------------------Imports----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as func

#---------------------------------VAE------------------------------------------

class VAE(nn.Module):
    '''
    VAE class.
    Architecture:

    	Encoder:
        	Two fully-connected layers fc1, fc2 with RelU.
        	Compresssing layers fc31, fc32.

		Decoder:
        	Three fully-connected layers fc4, fc5, fc6 with RelU
        	and sigmoid for last layer.
    '''

    #Initialize model
    def __init__(self,):
        super(VAE, self).__init__()

        self.fc1= nn.Linear(196,256)
        self.fc2= nn.Linear(256,64)
        self.fc31= nn.Linear(64,5)
        self.fc32= nn.Linear(64,5)

        self.fc4= nn.Linear(5,64)
        self.fc5= nn.Linear(64,256)
        self.fc6= nn.Linear(256,196)

    #Encoder function
    def encode(self, x):
        h= func.relu(self.fc1(x))
        z= func.relu(self.fc2(h))
        return self.fc31(z), self.fc32(z)

    #Reparametrize
    def reparameterize(self, mu, logsig):
        std = torch.exp(0.5*logsig)
        eps = torch.randn_like(std)
        return mu + eps*std

    #Decoder function
    def decode(self, x):
        h= func.relu(self.fc4(x))
        z= func.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(z))

    #Forward
    def forward(self, x):
        mu, logsig = self.encode(x.view(-1, 196))
        z = self.reparameterize(mu, logsig)
        return self.decode(z), mu, logsig