import numpy as np
import torch
from options import EvalOption

def get_accuracy(y_pred, y):
    # y_pred, y : tensor
    chosen_num = EvalOption.chosen_num
    cand_num = EvalOption.cand_num

    acc_num = 0
    sum_num = 0
    sort_outputs, idx = torch.sort(y_pred, descending=True)
    target = torch.gather(y, 1, idx).cuda()
    topk_label, _ = target.split([chosen_num, cand_num - chosen_num], dim=1)
    acc_num = acc_num + torch.sum(topk_label).item()
    sum_num = sum_num + y_pred.size(0) * chosen_num
    print('acc_num:', acc_num, 'sum_num:', sum_num)

    return acc_num/sum_num
