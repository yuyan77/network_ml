import re
import string

def parse_config_remove_random(config_dict, words, features_type):

    for f in config_dict:
        if f not in features_type.keys():
            continue
        if features_type[f] == 'random':
            continue

        words.append(f)
        if features_type[f] == 'dict':
            parse_config_remove_random(config_dict[f], words, features_type)
        elif features_type[f] == 'list':
            for ls in config_dict[f]:
                parse_config_remove_random(ls, words, features_type)
        elif features_type[f] == 'ip':
            ips = re.split('[./]', config_dict[f])
            words.extend(ips)
        elif features_type[f] == 'mac':
            macs = re.split(':', config_dict[f])
            words.extend(macs)
        else:
            remove = str.maketrans('', '', string.punctuation)
            without_punctuation = str(config_dict[f]).translate(remove)
            words.append(without_punctuation)


def parse_config_keep_all(config_dict, words, features_type):

    for f in config_dict:
        if f not in features_type.keys():
            continue
        words.append(f)
        if features_type[f] == 'dict':
            parse_config_keep_all(config_dict[f], words, features_type)
        elif features_type[f] == 'list':
            for ls in config_dict[f]:
                parse_config_keep_all(ls, words, features_type)
        elif features_type[f] == 'ip':
            ips = re.split('[./]', config_dict[f])
            words.extend(ips)
        elif features_type[f] == 'mac':
            macs = re.split(':', config_dict[f])
            words.extend(macs)
        else:
            remove = str.maketrans('', '', string.punctuation)
            without_punctuation = str(config_dict[f]).translate(remove)
            words.append(without_punctuation)


def parse_config_random_apart(config_dict, random_words, text_words, features_type):

    for f in config_dict:
        if f not in features_type.keys():
            continue
        text_words.append(f)
        if features_type[f] == 'random':
            random_words.append(config_dict[f])
        elif features_type[f] == 'dict':
            parse_config_random_apart(config_dict[f], random_words, text_words, features_type)
        elif features_type[f] == 'list':
            for ls in config_dict[f]:
                parse_config_random_apart(ls, random_words, text_words, features_type)
        elif features_type[f] == 'ip':
            ips = re.split('[./]', config_dict[f])
            text_words.extend(ips)
        elif features_type[f] == 'mac':
            macs = re.split(':', config_dict[f])
            text_words.extend(macs)
        else:
            remove = str.maketrans('', '', string.punctuation)
            without_punctuation = str(config_dict[f]).translate(remove)
            text_words.append(without_punctuation)
