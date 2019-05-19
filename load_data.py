import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing import shuffle
from constants import *

import json


def get_rsdd_data(start_index=0, end_index=10, set_="train"):
    if set_ == "train":
        filename = "training\\training"
    elif set_ == "validation":
        filename = "validation\\validation"
    elif set_ == "test":
        filename = "test\\test"
    else:
        print("unknown set name ", set_)
        return

    with open(RSDD_DATA_DIR + filename) as json_file:
        data = json_file.readlines()[start_index:end_index]
        data = list(map(json.loads, data))
    x_str = []
    y = []
    print(data[0])
    for row in data:
        for dict_ in row:
            if dict_["label"] == "control":
                label = 0
            elif dict_["label"] == "depression":
                label = 1
            else:
                label = 2
                print(dict_)
            if label != 2:
                for post in dict_["posts"]:
                    x_str.append(post[1])
                    y.append(label)
    print(len(x_str))
    return x_str, y


def get_smhd_data(start_index=0, end_index=10, set_="train"):
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
    x_str = []
    y = []
    labels = set()
    print("0 ", data[0])
    for dict_ in data:
        for label_str in dict_["label"]:
            labels.add(label_str)

            if label_str == "control":
                label = 0
            elif label_str == "depression":
                label = 1
            elif label_str == "bipolar":
                label = 3
            elif label_str == "anxiety":
                label = 4
            else:
                label = 2
                print(dict_)
            if label != 2:
                for post in dict_["posts"]:
                    x_str.append(post["text"])
                    y.append(label)

    print(labels)
    print(len(x_str))
    return x_str, y


def get_depression_data(start_index=0, end_index=1000, test_size=200):
    if end_index != 0:
        x_train_positive = open("train_positive.txt", "r", encoding="utf8").readlines()[start_index:end_index]
        x_train_negative = open("train_negative.txt", "r", encoding="utf8").readlines()[start_index:end_index]

        x_train_positive = np.asarray(x_train_positive, dtype='<U1000')
        x_train_negative = np.asarray(x_train_negative, dtype='<U1000')
        y_train_positive = np.ones_like(x_train_positive, dtype=np.uint16)
        y_train_negative = np.zeros_like(x_train_negative, dtype=np.uint16)

        print('train positive: ', y_train_positive.shape)
        print('train negative: ', y_train_negative.shape)
        train_ratio = y_train_positive.shape[0] / (y_train_positive.shape[0] + y_train_negative.shape[0])
        print('train positive percentage: ', train_ratio)

        x_train = np.concatenate((x_train_positive, x_train_negative), axis=0)
        y_train = np.concatenate((y_train_positive, y_train_negative), axis=0)

        x_train, y_train = shuffle(x_train, y_train)

        if test_size == 0:
            return x_train, y_train

    x_test_positive = open("test_positive.txt", "r", encoding="utf8").readlines()[:test_size]
    x_test_negative = open("test_negative.txt", "r", encoding="utf8").readlines()[:test_size]

    x_test_positive = np.asarray(x_test_positive, dtype='<U1000')
    x_test_negative = np.asarray(x_test_negative, dtype='<U1000')

    y_test_positive = np.ones_like(x_test_positive, dtype=np.uint16)
    y_test_negative = np.zeros_like(x_test_negative, dtype=np.uint16)
    print('test positive: ', y_test_positive.shape)
    print('test negative: ', y_test_negative.shape)

    test_ratio = y_test_positive.shape[0] / (y_test_positive.shape[0] + y_test_negative.shape[0])
    print('test positive percentage: ', test_ratio)

    x_test = np.concatenate((x_test_positive, x_test_negative), axis=0)
    y_test = np.concatenate((y_test_positive, y_test_negative), axis=0)

    if end_index == 0:
        return x_test, y_test

    return x_train, x_test, y_train, y_test


def get_bipolar_disorder_data(start_index=0, skiprows_start=10 ** 3, skiprows_end=10 ** 7, nrows=2 * 10 ** 3,
                              test_size=0.2):
    df = pd.read_csv(BIPOLAR_DATA_DIR + "bipolar_control_reddit.csv",
                     skiprows=list(range(1, start_index)) + list(range(skiprows_start, skiprows_end)), nrows=nrows)
    print(df.head())
    df = df[["body", "classification"]]
    df = df.dropna(axis=0, how='any')
    x = df["body"].values
    y = df["classification"].values

    x, y = shuffle(x, y)

    if test_size > 1:
        test_size = test_size / nrows

    if test_size == 0 or test_size == 1:
        return x, y

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_train_, x_test_, y_train_, y_test_ = get_bipolar_disorder_data()
    print(x_train_[0])
    print(x_train_.shape)
    print(x_test_.shape)
    print(y_train_.shape)
    print(y_test_.shape)
