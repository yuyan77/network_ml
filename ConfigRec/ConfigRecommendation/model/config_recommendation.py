import torch.nn as nn
from model.config_encoder.config_encoder_v2 import ConfigEncoder
from model.vm_encoder import vm_encoder_v2
from model.prediction import prediction_v2


class ConfigRec(nn.Module):
    def __init__(self, words_num_in_dict, word_num, query_size, embedding_size, nhead, hidden_size):
        super(ConfigRec,self).__init__()
        self.embedding_size = embedding_size
        self.query_size = query_size
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.config_encoder = ConfigEncoder(words_num_in_dict, query_size, embedding_size, nhead, hidden_size)
        self.vm_encoder = vm_encoder_v2.VMEncoder(query_size, embedding_size, hidden_size)
        self.prediction = prediction_v2.Prediction(embedding_size)

    def forward(self, request_input, candidate_inputs, history_inputs, vm_id):
        batch_size, cand_num, word_num = candidate_inputs.size()
        _, his_num, _ = history_inputs.size()

        requst_config = self.config_encoder(request_input, vm_id)
        history_configs = self.config_encoder(history_inputs.view(batch_size*his_num, word_num), vm_id)
        history_configs = history_configs.view(batch_size, his_num, self.embedding_size)
        vm = self.vm_encoder(history_configs, vm_id)

        candidate_configs = self.config_encoder(candidate_inputs.view(batch_size*cand_num, word_num), vm_id)
        candidate_configs = candidate_configs.view(batch_size, cand_num, self.embedding_size)
        output = self.prediction(vm, requst_config, candidate_configs)

        return output
