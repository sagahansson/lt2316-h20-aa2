import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_D, hid, out_D, num_layers):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.hid = hid
        self.gru = nn.GRU(in_D, hid, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hid, out_D)
    
    def forward(self, batch, device):
        
        out, h_out = self.gru(batch)
        #pred = self.fc(h_out[-1])
        pred = self.fc(out)
        
        return pred