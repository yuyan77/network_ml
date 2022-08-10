import os
import numpy as np
import random
import math
from tqdm import tqdm
import json
import csv
import re
import nltk
import json
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import string
import torch
import torch.nn as nn
from options import RandomOption, TrainOption
from feature.word2vec import bags_of_words

nltk.download('punkt')


def wordpiece(sentence):
    '''
    convert a sentence to several useful words, and remove all punctuations
    :param sentence:
    :return: set of words
    '''
    remove = str.maketrans('', '', string.punctuation)
    without_punctuation = sentence.translate(remove)
    words = nltk.word_tokenize(without_punctuation)
    return words


def create_random_data(data_num):
    '''
    create random train data for test
    :param data_num: the number of dataset
    :return: request_input, history_inputs, candidate_inputs, vm_id, label
    '''
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


def save_words(words, save_file):
    '''
    write to csv file
    format:
    line : a set of config data
    words = [request_words, [history_words], [candidate_words], vm id, label]
    :param words:
    :param save_file:
    :return:
    '''
    with open(save_file, "a+", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(words)


def initial_part_words(raw_data, save_file):
    '''
    initial processing, store in file
    for a VM at a time:
    every set of model input includes: request_input, history_inputs, candidate_inputs, vm_id, label
    1. preprocess raw data, including data cleaning, form sets of datas at the granularity of VM
    2. process request config,
    3. process candidate configs (stored in database and may be delivered)
    4. process history configs of a similar VM (arranged in time order, the number of history configs is unified and adjustable.)
    5. get vm id
    6. get label :0/1, whether these candidate configs are selected to deliver
    :return:
    '''

    # preprocess

    # process configs ( divide every config into useful words)
    request_config_words = wordpiece()
    history_configs_words = wordpiece()
    candidate_configs_words = wordpiece()

    # get vm id words
    vm_words =

    # label representation
    label =

    # save words
    save_file([request_config_words, history_configs_words, candidate_configs_words, vm_words, label], save_file)


def initial_part_words_without_label(raw_data, save_file):
    '''
    for a VM at a time:
    every set of model input includes: request_input, history_inputs, candidate_inputs, vm_id, label
    1. preprocess raw data, including data cleaning, form sets of datas at the granularity of VM
    2. process request config,
    3. process candidate configs (stored in database and may be delivered)
    4. process history configs of a similar VM (arranged in time order, the number of history configs is unified and adjustable.)
    5. get vm id
    :return:
    '''

    # preprocess


    # process configs ( divide every config into useful words)
    request_config_words =
    history_configs_words =
    candidate_configs_words =

    # get vm id words
    vm_words =

    # save words
    save_file([request_config_words, history_configs_words, candidate_configs_words, vm_words], save_file)


def get_train_inputs(train_file):
    '''
    preprocess real config data as train data, every config is divided into words
    including request_input, history_inputs, candidate_inputs, vm_id, label
    used to train、eval、test
    :param config_file:  raw data of train config
    :return:
    '''

    all_sets = []
    with open(train_file, 'r') as f:
        reader = csv.reader(f)
        for s in reader:
            all_sets.append(s)

    # all set
    all_words = []
    request_configs_words = []
    history_configs_words = []
    candidate_configs_words = []
    vms_words = []
    labels = []

    # establish word2id dictionary and convert every config (words) to one vector
    word2id_dict = bags_of_words.get_word2id_dict(all_words)
    request_input = bags_of_words.convert_to_vec(request_configs_words, word2id_dict, TrainOption.max_words_num)

    history_inputs = []
    for config_words in history_configs_words:
        input = bags_of_words.convert_to_vec(config_words, word2id_dict, TrainOption.max_words_num)
        history_inputs.append(input)

    candidate_inputs = []
    for config_words in candidate_configs_words:
        input = bags_of_words.convert_to_vec(config_words, word2id_dict, TrainOption.max_words_num)
        candidate_inputs.append(input)

    # convert vms words to vector
    vm_id = []

    # process labels
    label = []

    return request_input, history_inputs, candidate_inputs, vm_id, label


def get_test_inputs(test_file):
    '''
    preprocess real config data as eval data, every config is divided into words
    including request_input, history_inputs, candidate_inputs, vm_id
    used to predict
    :param config_file: raw data of train config
    :return:
    '''
    all_sets = []
    with open(test_file, 'r') as f:
        reader = csv.reader(f)
        for s in reader:
            all_sets.append(s)

    # all set
    all_words = []
    request_configs_words = []
    history_configs_words = []
    candidate_configs_words = []
    vms_words = []

    # establish word2id dictionary and convert every config (words) to one vector
    word2id_dict = bags_of_words.get_word2id_dict(all_words)
    request_input = bags_of_words.convert_to_vec(request_configs_words, word2id_dict, TrainOption.max_words_num)

    history_inputs = []
    for config_words in history_configs_words:
        input = bags_of_words.convert_to_vec(config_words, word2id_dict, TrainOption.max_words_num)
        history_inputs.append(input)

    candidate_inputs = []
    for config_words in candidate_configs_words:
        input = bags_of_words.convert_to_vec(config_words, word2id_dict, TrainOption.max_words_num)
        candidate_inputs.append(input)

    # convert vms words to vector
    vm_id = []

    return request_input, history_inputs, candidate_inputs, vm_id



