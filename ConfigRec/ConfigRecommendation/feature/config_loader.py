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

nltk.download('punkt')


def load_config(filename):
    file_path = filename+'.json'
    result = open(file_path, 'r', encoding="utf-8")
    data = json.load(result)
    dict = json.loads(data)
    config = dict[filename]
    config_data = config[0]
    return config_data


def convert_to_vec(words, word2id, max_words_num):
    vec = []
    for word in words:
        try:
            vec.append(word2id[word])
        except KeyError:
            print("word not found in word2id!")

    if len(vec) < max_words_num:
        vec += [0] * (max_words_num - len(vec))
    else:
        vec = vec[:max_words_num]

    return vec


def write_feature_type():
    features_dict={}
    random_list=['id', 'project_id', 'tenant_id', 'network_id', 'gatewayPortId', 'gatewayportid', 'ipV4_rangeId', 'dns_publish_fixed_ip']
    dict_list = ['gateway_port_detail']
    list_list = ['allocation_pools', 'host_routes']
    text_list = ['name', 'enable_dhcp', 'revision_number', 'dns_publish_fixed_ip', 'use_default_subnet_pool']
    ip_list = ['cidr', 'gateway_ip', 'start', 'end', 'destination', 'nexthop']
    mac_list = ['gateway_macAddress']
    for r in random_list:
        features_dict[r] = 'random'
    for l in list_list:
        features_dict[l] = 'list'
    for d in dict_list:
        features_dict[d] = 'dict'
    for t in text_list:
        features_dict[t] = 'text'
    for i in ip_list:
        features_dict[i] = 'ip'
    for m in mac_list:
        features_dict[m] = 'mac'

    with open("../data/subnets_feature_type.json", "w") as f:
        f.write(json.dumps(features_dict, indent=4))


def read_feature_type(filename):
    with open(filename+'_feature_type.json') as f:
        output_dict = json.loads(f.read())
    return output_dict


