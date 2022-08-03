import numpy as np
import random
import math
from tqdm import tqdm
import json
import re
import nltk
import json
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import string
import torch
import torch.nn as nn
from options import RandomOption

nltk.download('punkt')


def wordpiece(config):
    remove = str.maketrans('', '', string.punctuation)
    without_punctuation = config.translate(remove)
    word = nltk.word_tokenize(without_punctuation)
    return word


def create_random_data(data_num):
    request_input = torch.randint(0, RandomOption.words_num_in_dict, (data_num, RandomOption.max_words_num))
    history_inputs = torch.randint(0, RandomOption.words_num_in_dict, (data_num, RandomOption.his_num, RandomOption.max_words_num))
    candidate_inputs = torch.randint(0, RandomOption.words_num_in_dict, (data_num, RandomOption.cand_num, RandomOption.max_words_num))
    vm_id = torch.randn(data_num, RandomOption.query_size)
    print("candidate inputs:", candidate_inputs.shape)
    print("history inputs:", history_inputs.shape)
    print("request input:", request_input.shape)
    print("vm id:", vm_id.shape)

    label = torch.Tensor([1, 1, 0, 0, 0]).view(1, RandomOption.cand_num)
    x = torch.Tensor([1, 1, 0, 0, 0]).view(1, RandomOption.cand_num)
    for i in range(data_num - 1):
        label = torch.cat((label, x), dim=0)

    print("label:", label.shape)

    return request_input, history_inputs, candidate_inputs, vm_id, label


def process_train_data(config_file):
    with open(config_file, 'r'):
        data = 0

    return data


def process_eval_data(config_file):
    with open(config_file, 'r'):
        data = 0

    return data
