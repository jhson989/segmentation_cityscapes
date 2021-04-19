
import torch
import torch.nn as nn

class FCN8S(nn.Module):

    def __init__(self, args):
        super(FCN32, self).__init__()
        self.num_class = args.num_class

    def forward(self, x):

