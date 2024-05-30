import numpy as np
import torch.nn.functional as F

import copy
import json
import yaml


############ IO ################
def load_json(json_file):
    data = None
    with open(json_file, "r") as fin:
        data = json.load(fin)

    return data


def load_txt(file):
    with open(file) as f:
        return f.read().strip()


def write_txt(file, content):
    with open(file, "w") as f:
        f.write(content)


def dump_json(json_data, output_dir):
    with open(output_dir, "w") as fout:
        fout.write(json.dumps(json_data, indent=2))


def load_yaml(yaml_file):
    config = None
    with open(yaml_file, "r") as fin:
        config = yaml.safe_load(fin)

    return config


def save_npz(data, output_dir, compressed=False):
    if compressed:
        np.savez_compressed(output_dir, **data)
    else:
        np.savez(output_dir, **data)


def load_npz(data_file, allow_pickle=True):
    return np.load(data_file, allow_pickle=allow_pickle)


def mkdir(path):
    path.mkdir(parents=True, exist_ok=True)


def print_config(config, depth=0):
    config_str = ""
    for k, v in config.items():
        if isinstance(v, dict):
            config_str += "{}* {}\n:".format("  " * depth, k)
            config_str += print_config(v, depth + 1)
        else:
            config_str += "{}* {}: {}\n".format("  " * depth, k, v)

    return config_str


def extend_list(list_to_extend, target_len):
    while 2 * len(list_to_extend) < target_len:
        list_to_extend += list_to_extend

    if len(list_to_extend) >= target_len:
        return list_to_extend[0:target_len]
    else:
        return list_to_extend + list_to_extend[0 : target_len - len(list_to_extend)]


def merge_number_dict(mydict, dict_to_merge):
    for k, v in dict_to_merge.items():
        if k not in mydict:
            # Important to use deep copy!
            mydict[k] = copy.deepcopy(v)
        elif isinstance(v, dict):
            merge_number_dict(mydict[k], v)
        else:
            mydict[k] += v


def subtract_number_dict(mydict, dict_to_subtract):
    for k, v in dict_to_subtract.items():
        assert k in mydict, "Can't substract dict!"
        if isinstance(v, dict):
            subtract_number_dict(mydict[k], v)
        else:
            mydict[k] -= v
