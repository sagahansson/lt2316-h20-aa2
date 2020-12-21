import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_D, hid, out_D, num_layers, hid_state):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.hid = hid
        self.hid_state = hid_state
        self.gru = nn.GRU(in_D, hid, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hid, out_D)
    
    def forward(self, batch, device):
        out, h_out = self.gru(batch)
        if self.hid_state:
            pred = self.fc(h_out[-1]) # dim [batch_size, out_D]
        else:
            pred = self.fc(out) # dim [batch_size, max_seq_len, out_D]
        return pred