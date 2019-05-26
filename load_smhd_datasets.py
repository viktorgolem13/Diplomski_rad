from constants import *
import json
import numpy as np
from preprocessing import shuffle


def get_smhd_data(start_index=0, end_index=100, set_="train"):
    if set_ == "train":
        filename = "train.jl"
    elif set_ == "validation":
        filename = "dev.jl"
    elif set_ == "test":
        filename = "test.jl"
    else:
        print("unknown set name ", set_)
        return

    with open(SMHD_DATA_DIR + filename) as json_file:
        data = json_file.readlines()[start_index:end_index]
        data = list(map(json.loads, data))
    x0 = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    labels = set()
    print("0 ", data[0])
    for dict_ in data:
        for label_str in dict_["label"]:
            labels.add(label_str)

            if label_str == "control":
                for post in dict_["posts"]:
                    if len(post["text"]) > 1:
                        x0.append(post["text"])
            elif label_str == "depression":
                for post in dict_["posts"]:
                    if len(post["text"]) > 1:
                        x1.append(post["text"])
            elif label_str == "bipolar":
                for post in dict_["posts"]:
                    if len(post["text"]) > 1:
                        x2.append(post["text"])
            elif label_str == "anxiety":
                for post in dict_["posts"]:
                    if len(post["text"]) > 1:
                        x3.append(post["text"])
            else:
                for post in dict_["posts"]:
                    if len(post["text"]) > 1:
                        x4.append(post["text"])

    print(labels)
    return x0, x1, x2, x3, x4


def add_user(class_list, posts):
    temp = ""
    for post in posts:
        if len(post["text"]) > 1:
            temp += post["text"]
    if len(temp) > 1:
        class_list.append(temp)


def get_smhd_data_user_level(start_index=0, end_index=10, set_="train"):
    if set_ == "train":
        filename = "train.jl"
    elif set_ == "validation":
        filename = "dev.jl"
    elif set_ == "test":
        filename = "test.jl"
    else:
        print("unknown set name ", set_)
        return

    with open(SMHD_DATA_DIR + filename) as json_file:
        if end_index is None:
            data = json_file.readlines()[start_index:]
        else:
            data = json_file.readlines()[start_index:end_index]
        data = list(map(json.loads, data))
    x0 = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    labels = set()
    print("0 ", data[0])
    for dict_ in data:
        for label_str in dict_["label"]:
            labels.add(label_str)
            if label_str == "control":
                add_user(x0, dict_["posts"])
            elif label_str == "depression":
                add_user(x1, dict_["posts"])
            elif label_str == "bipolar":
                add_user(x2, dict_["posts"])
            elif label_str == "anxiety":
                add_user(x3, dict_["posts"])
            else:
                add_user(x4, dict_["posts"])

    print(labels)
    return x0, x1, x2, x3, x4


def generate_labels(x0, x1):
    y0 = np.zeros(len(x0), dtype=np.int16)
    y1 = np.ones(len(x1), dtype=np.int16)

    return y0, y1


def prepare_binary_data(x0, x1):
    y0, y1 = generate_labels(x0, x1)

    x = np.concatenate((x0, x1), axis=0)
    y = np.concatenate((y0, y1), axis=0)

    return shuffle(x, y)
