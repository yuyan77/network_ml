import torch
import torch.nn as nn

"""
Prediction Model:
for every candidate config (Config Encoder), predict a score through the history config sequence of similar VM (VM Encoder)

for every candidate config of a VM:
input size:
candidate_inputs = [batch_size, cand_num, embedding_size], cand_num is the num of candidate configs of one VM
vm_input = [batch_size, 2 * embedding_size], vm_input = vm vector + request config vector, request config is the config being requested

output size:
output = [batch_size, cand_num], every score of a candidate config predicts the possibility the to be chosen
"""

class Prediction(nn.Module):
    def __init__(self, embedding_size):
        super(Prediction, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(embedding_size*2, embedding_size, bias=False),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size, bias=False)
        )

    def forward(self, vm_input, request_input, candidate_inputs):

        batch_size, cand_num, embedding_size = candidate_inputs.size()
        vm_requst = torch.cat((vm_input, request_input), dim=1)
        output_ffn = self.ffn(vm_requst)
        scores = torch.bmm(candidate_inputs, output_ffn.view(batch_size, embedding_size, 1)).view(batch_size, cand_num)
        output = nn.Softmax(dim=1)(scores)
        return output
