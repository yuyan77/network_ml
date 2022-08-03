import torch
import torch.nn as nn
import random
import math

"""
module of NPA

input size:
config_input = [batch_size, num, embedding_size]
vm_id = [batch_size, query_size], vm_id is a vector which identifies a VM

output size:
output = [batch_size, embedding_size]
"""

class PersonalizedAttention(nn.Module):
    def __init__(self, query_size, embedding_size):

        super(PersonalizedAttention, self).__init__()
        self.w_query=nn.Sequential(nn.Linear(query_size, embedding_size, bias=True),
                                   nn.Tanh()
                                   )

    def forward(self, config_input, vm_id):

        batch_size, _ = vm_id.size()
        batch_num, word_num, embedding_size = config_input.size()
        query = self.w_query(vm_id)
        if batch_num > batch_size:
            num = (int)(batch_num / batch_size)
            query = torch.repeat_interleave(query, repeats=num, dim=0)
        scores = torch.matmul(config_input, query.view(batch_num, embedding_size, 1)).view(batch_num, word_num)
        scores_sm = nn.Softmax(dim=1)(scores)
        output = torch.bmm(scores_sm.view(batch_num, 1, word_num), config_input)
        output = output.view(batch_num, embedding_size)

        return output

