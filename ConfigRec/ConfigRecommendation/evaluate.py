import torch
import os
from tqdm import tqdm
from data.datasets.dataset import EvalDataSet
from torch.utils.data import DataLoader
import feature.data_preprocess as dp
import feature.config_loader as cl
from options import EvalOption

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(raw_file, eval_words_file):
    # write file
    if not os.path.exists("./data/eval_data/{}.csv".format(eval_words_file)):
        dp.initial_part_words("./data/raw_data/{}.txt".format(raw_file))

    request_input, history_inputs, candidate_inputs, vm_id, label = dp.get_train_inputs(eval_words_file)

    batch_size = EvalOption.batch_size
    chosen_num = EvalOption.chosen_num
    cand_num = EvalOption.cand_num
    eval_dataset = EvalDataSet(request_input, history_inputs, candidate_inputs, vm_id, label)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 载入模型
    if not os.path.exists("./checkpoints/train_model.pth"):
        print("error! model not found")

    model = torch.load("./checkpoints/train_model.pth")

    acc_num = 0
    sum_num = 0

    for request_input, history_inputs, candidate_inputs, vm_id_input, label in tqdm(eval_loader):
        request_input, history_inputs, candidate_inputs, vm_id_input, label = request_input.cuda(), history_inputs.cuda(), candidate_inputs.cuda(), vm_id_input.cuda(), label.cuda()

        outputs = model(request_input, candidate_inputs, history_inputs, vm_id_input)
        sort_outputs ,idx = torch.sort(outputs, descending=True)
        target = torch.gather(label, 1, idx).cuda(  )
        topk_label, _ = target.split([chosen_num, cand_num - chosen_num], dim=1)
        acc_num = acc_num + torch.sum(topk_label).item()
        sum_num = sum_num + outputs.size(0) * chosen_num
        print('acc_num:', acc_num, 'sum_num:', sum_num)

    return acc_num/sum_num


if __name__ == '__main__':
    batch_size = 64
    chosen_num = 2
    cand_num = 5
    print(evaluate())

