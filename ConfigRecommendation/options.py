import os

class DataOption():

    max_words_num = 50
    words_num_in_dict = 200

    cand_num = 5
    his_num = 6
    chosen_num = 2


class ModelOption():
    '''
    General configurations applied to all models
    '''
    # for train
    batch_size = 16
    learning_rate = 0.001

    # for embedding
    embedding_size = 64

    # for model
    query_size = 30
    hidden_size = 50
    nhead = 2


class RandomOption(DataOption, ModelOption):
    words_num_in_dict = 100


class TrainOption(DataOption, ModelOption):
    epoch = 4
    batch_size = 64


class EvalOption(DataOption, ModelOption):
    batch_size = 16


if __name__ == '__main__':
    print(TrainOption.batch_size)
    print(ModelOption.batch_size)