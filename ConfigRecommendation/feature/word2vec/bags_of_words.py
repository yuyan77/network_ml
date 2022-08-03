from collections import Counter
import numpy as np


def get_word_dict(words):
    counter = Counter(words)
    word_dict = {}
    n = np.arange(1, len(counter) + 1)
    for i, f in zip(n, counter):
        word_dict[f] = i
    return word_dict


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
