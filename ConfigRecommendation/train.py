import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from data.datasets.dataset import ConfigDataSet
from model.config_recommendation import ConfigRec
import feature.data_preprocess as dp
import feature.config_loader as cl
from options import TrainOption
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(train_file=None, config_file=None):

    if os.path.exists("./data/train_data/{}.txt".format(train_file)):
        request_input, history_inputs, candidate_inputs, vm_id, label = cl.load_train_data(train_file)
    else:
        request_input, history_inputs, candidate_inputs, vm_id, label = dp.process_train_data(config_file)

    config_dataset = ConfigDataSet(request_input, history_inputs, candidate_inputs, vm_id, label)
    config_num = config_dataset.__len__()
    train_size = int(config_num * 0.8)
    valid_size = config_num - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(config_dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=TrainOption.batch_size, shuffle=False,drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=TrainOption.batch_size, shuffle=False,drop_last=True)

    model = ConfigRec(TrainOption.words_num_in_dict,
                      TrainOption.max_words_num,
                      TrainOption.query_size,
                      TrainOption.embedding_size,
                      TrainOption.nhead,
                      TrainOption.hidden_size).cuda()
    criterion = nn.MSELoss(reduce=True, size_average=True)
    optimizer = optim.Adam(model.parameters(), TrainOption.learning_rate)
    for i in range(TrainOption.epoch):
        train_loss =[]
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
            request_input, history_inputs, candidate_inputs, vm_id_input, label=request_input.cuda(), history_inputs.cuda(), candidate_inputs.cuda(), vm_id_input.cuda(), label.cuda()

            outputs = model(request_input, history_inputs, candidate_inputs, vm_id_input)
            loss = criterion(outputs, label)
            valid_loss.append(loss.item())

        print('epoch ', i+1, ' train_loss=', np.mean(train_loss))
        print('epoch ', i+1, ' valid_loss=', np.mean(valid_loss))

    torch.save(model, "./checkpoints/train_model.pth")


if __name__ == '__main__':
    train()

