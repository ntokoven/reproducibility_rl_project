import torch
from torch import nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)
        self.ReLU = nn.ReLU()
        self.LogSoftmax = torch.nn.LogSoftmax()
        
    def forward(self, x):
        out1 = self.ReLU(self.l1(x))
        out2 = self.l2(out1)
        return self.LogSoftmax(out2)
