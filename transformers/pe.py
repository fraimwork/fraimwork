import torch
import math
import torch.nn as nn
from math import exp, cos, sin, log

class PositionalEncoding(nn.Module):
    '''
    We implement the positional encoding as alternating sin and cosine functions.
    In the __init__ function, you simply set the variable pe according the original paper.
    In the forward function, you add the pe to the input
    '''
    def __init__(self, d_model, max_seq_length):
        '''
        Initialize the positional encoding as alternating sin and cosine functions of the dimension position and the model size d_model
        Input:
            d_model (int) - dimension of the Transformer
            max_seq_length (int) - maximum sequence length of the Transformer
        '''
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)

        for pos in range(max_seq_length):
            for i in torch.arange(0, d_model, 2):
                pe[pos, i] = sin(pos * exp(i * (-log(10000)/d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = cos(pos * exp(i * (-log(10000)/d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """Add positional encoding (after appropriate slicing) to the input x
        Input:
            x (torch.Tensor) - Tensor of size B x T x d_model.
        Output:
            torch.Tensor - input with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]