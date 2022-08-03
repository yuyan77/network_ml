import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import random
import math


class ConfigDataSet(Data.Dataset):
    def __init__(self, request_input, history_inputs, candidate_inputs, vm_id, label):

        super(ConfigDataSet, self).__init__()
        self.candidata_inputs = candidate_inputs
        self.request_input = request_input
        self.history_inputs = history_inputs
        self.vm_id = vm_id
        self.label = label

    def __len__(self):
        return self.request_input.shape[0]

    def __getitem__(self, idx):
        return self.request_input[idx], self.history_inputs[idx], \
               self.candidata_inputs[idx], self.vm_id[idx], \
               self.label[idx]


class TrainDataSet(Data.Dataset):
    def __init__(self, request_input, history_inputs, candidate_inputs, vm_id, label):
        super(TrainDataSet, self).__init__()
        self.candidata_inputs = candidate_inputs
        self.request_input = request_input
        self.history_inputs = history_inputs
        self.vm_id = vm_id
        self.label = label

    def __len__(self):
        return self.request_input.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.request_input[idx])).float(), \
               torch.from_numpy(np.array(self.history_inputs[idx])).float(),\
               torch.from_numpy(np.array(self.candidate_inputs[idx])).float(), \
               torch.from_numpy(np.array(self.vm_id[idx])).float(), \
               torch.from_numpy(np.array(self.label[idx])).float()


class EvalDataSet(Data.Dataset):
    def __init__(self, request_input, history_inputs, candidate_inputs, vm_in_input, label):

        super(EvalDataSet, self).__init__()
        self.candidata_inputs = candidate_inputs
        self.request_input = request_input
        self.history_inputs = history_inputs
        self.vm_id_input = vm_in_input
        self.label = label

    def __len__(self):
        return self.request_input.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.request_input[idx])).float(), \
               torch.from_numpy(np.array(self.history_inputs[idx])).float(),\
               torch.from_numpy(np.array(self.candidate_inputs[idx])).float(), \
               torch.from_numpy(np.array(self.vm_id_input[idx])).float(), \
               torch.from_numpy(np.array(self.label[idx])).float()


class TestDataSet(Data.Dataset):
    def __init__(self, request_input, history_inputs, candidate_inputs, vm_in_input):

        super(TestDataSet, self).__init__()
        self.candidata_inputs = candidate_inputs
        self.request_input = request_input
        self.history_inputs = history_inputs
        self.vm_id_input = vm_in_input

    def __len__(self):
        return self.request_input.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.request_input[idx])).float(), \
               torch.from_numpy(np.array(self.history_inputs[idx])).float(), \
               torch.from_numpy(np.array(self.candidate_inputs[idx])).float(), \
               torch.from_numpy(np.array(self.vm_id_input[idx])).float()


