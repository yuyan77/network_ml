import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
from tqdm import tqdm
from data.datasets.dataset import ConfigDataSet
from model.config_recommendation import ConfigRec
import feature.data_preprocess as dp
from options import RandomOption

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_random(random_data_num):
    # random data test
    request_input, history_inputs, candidate_inputs, vm_id, label = dp.create_random_data(random_data_num)
    config_dataset = ConfigDataSet(request_input, history_inputs, candidate_inputs, vm_id, label)
    config_num = config_dataset.__len__()
    train_size = int(config_num * 0.8)
    valid_size = config_num - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(config_dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=RandomOption.batch_size, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=RandomOption.batch_size, shuffle=False, drop_last=True)

    model = ConfigRec(RandomOption.words_num_in_dict,
                      RandomOption.max_words_num,
                      RandomOption.query_size,
                      RandomOption.embedding_size,
                      RandomOption.nhead,
                      RandomOption.hidden_size).cuda()
    criterion = nn.MSELoss(reduce=True, size_average=True)
    optimizer = optim.Adam(model.parameters(), RandomOption.learning_rate)
    for i in range(RandomOption.epoch):
        train_loss = []
        for request_input, history_inputs, candidate_inputs, vm_id_input, label in tqdm(train_loader):
            request_input, history_inputs, candidate_inputs, vm_id_input, label = request_input.cuda(), history_inputs.cuda(), \
                                                                                  candidate_inputs.cuda(), vm_id_input.cuda(), \
                                                                                  label.cuda()

            outputs = model(request_input, candidate_inputs, history_inputs, vm_id_input)
            print("outputs: ", outputs.size())
            loss = criterion(outputs.float(), label.float())
            print("loss: ", loss)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        valid_loss = []
        for request_input, history_inputs, candidate_inputs, vm_id_input, label in tqdm(valid_loader):
            request_input, history_inputs, candidate_inputs, vm_id_input, label = request_input.cuda(), history_inputs.cuda(), candidate_inputs.cuda(), vm_id_input.cuda(), label.cuda()

            outputs = model(request_input, history_inputs, candidate_inputs, vm_id_input)
            loss = criterion(outputs, label)
            valid_loss.append(loss.item())

        print('epoch ', i + 1, ' train_loss=', np.mean(train_loss))
        print('epoch ', i + 1, ' valid_loss=', np.mean(valid_loss))

    acc_num = 0
    sum_num = 0
    batch_size = RandomOption.batch_size
    chosen_num = RandomOption.chosen_num
    cand_num = RandomOption.cand_num
    for request_input, history_inputs, candidate_inputs, vm_id_input, label in tqdm(valid_loader):
        request_input, history_inputs, candidate_inputs, vm_id_input, label = request_input.cuda(), history_inputs.cuda(), candidate_inputs.cuda(), vm_id_input.cuda(), label.cuda()

        outputs = model(request_input, candidate_inputs, history_inputs, vm_id_input)
        sort_outputs, idx = torch.sort(outputs, descending=True)
        target = torch.gather(label, 1, idx).cuda()
        topk_label, _ = target.split([chosen_num, cand_num - chosen_num], dim=1)
        acc_num = acc_num + torch.sum(topk_label).item()
        sum_num = sum_num + outputs.size(0) * chosen_num
        print('acc_num:', acc_num, 'sum_num:', sum_num)


if __name__ == '__main__':
    test_random(64)


