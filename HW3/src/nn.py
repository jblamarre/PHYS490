# =============================================================================
#
# PHYS490 - Winter 2021 - HW3 (J. Lamarre)
#
# =============================================================================
import torch.nn as nn
import torch.nn.functional as func

#LSTM RNN with fully connected layer
class lstm_reg(nn.Module):
    def __init__(self, n_dim, n_hidden, n_layers, seq_len):
        super(lstm_reg, self).__init__()
        
        self.rnn = nn.LSTM(n_dim, n_hidden, n_layers, batch_first = True)
        self.fc = nn.Linear(n_hidden * seq_len, n_dim)
        
    def forward(self, x):
        x, _ = self.rnn(x)
        b, s, h = x.shape
        x = self.fc(x)
        return x
