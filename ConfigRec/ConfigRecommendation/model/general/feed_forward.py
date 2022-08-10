import torch
import torch.nn as nn
import random
import math

"""
module of transformer

input size:
config_input = [batch_size, embedding_size]

output size:
output = [batch_size, embedding_size]
"""

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size, bias=False)
        )
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, config_input):

        config_input = config_input.cuda()
        residual = config_input.cuda()
        ff = self.fc(config_input).cuda()
        output = self.layer_norm((ff + residual).cuda())
        return output
