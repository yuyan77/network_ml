import torch
import torch.nn as nn
from model.general.feed_forward import PoswiseFeedForwardNet
from model.general.personalized_attention import PersonalizedAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Config Encoder:
extract features from config data, and convert every config to vector

DL model used: embedding, multi-head attention, personalized attention, add&norm(transformer)

for every config:
input size:
config_input = [batch_size, max_words_num], max_words_num is the maximum length of words of every config, padding with 0
vm_id = [batch_size, query_size], vm_id is a vector which identifies a VM

output size:
output2 = [batch_size, embedding_size], every config is represented as one vector
"""

class ConfigEncoder(nn.Module):

    def __init__(self, words_num_in_dict, query_size, embedding_size, nhead, hidden_size):
        super(ConfigEncoder, self).__init__()

        # embedding
        self.embed = nn.Embedding(words_num_in_dict, embedding_size)

        # multi-head attention
        self.multihead_attn = nn.MultiheadAttention(embedding_size, nhead, batch_first=True)
        self.add_norm1 = PoswiseFeedForwardNet(embedding_size, hidden_size)

        # personalized attention
        self.person_attn = PersonalizedAttention(query_size, embedding_size)
        self.add_norm2 = PoswiseFeedForwardNet(embedding_size, hidden_size)

    def forward(self, config_input, vm_id):
        config_emb = self.embed(config_input.long()).float()
        config_attn1, _ = self.multihead_attn(config_emb, config_emb, config_emb)
        output1 = self.add_norm1(config_attn1)
        config_attn2 = self.person_attn(output1, vm_id)
        output2 = self.add_norm2(config_attn2)

        return output2

