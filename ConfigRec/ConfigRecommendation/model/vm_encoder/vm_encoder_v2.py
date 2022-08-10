import torch.nn as nn
from model.general.feed_forward import PoswiseFeedForwardNet
from model.general.personalized_attention import PersonalizedAttention

"""
VM Encoder:
extract features from config sequence of a VM, and convert the sequence to vector

DL model used: personalized attention, add&norm(transformer)

for every config sequence of a VM:
input size:
history inputs = [batch_size, his_num, embedding_size], his_num is the length of every config sequence
vm_id = [batch_size, query_size], vm_id is a vector which identifies a VM

output size:
output = [batch_size, embedding_size], every config sequence of a VM is represented as one vector
"""


class VMEncoder(nn.Module):
    def __init__(self, query_size, embedding_size, hidden_size):
        super(VMEncoder, self).__init__()

        self.person_attn = PersonalizedAttention(query_size, embedding_size)
        self.add_norm = PoswiseFeedForwardNet(embedding_size, hidden_size)

    def forward(self, history_inputs, vm_id):
        config_attn = self.person_attn(history_inputs, vm_id)
        output = self.add_norm(config_attn)

        return output
