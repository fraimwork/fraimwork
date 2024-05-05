from torch import nn
import torch
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, 1)

    def forward(self, inputs):
        energy = torch.tanh(self.W(inputs))
        attention_scores = F.softmax(self.v(energy), dim=1)
        weighted_inputs = attention_scores * inputs
        context_vector = torch.sum(weighted_inputs, dim=1)
        return context_vector