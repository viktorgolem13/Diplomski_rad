import numpy as np
from constants import *


def get_curr_x_y(x_train, y_train, batches_in_file, i):
    get_curr_x_y.curr_i = None
    get_curr_x_y.x_curr = None
    get_curr_x_y.y_curr = None

    if get_curr_x_y.curr_i != i:
        get_curr_x_y.x_curr = np.load(x_train + str(i) + ".npy")
        get_curr_x_y.y_curr = np.load(y_train + str(i) + ".npy")
        get_curr_x_y.curr_i = i

    if batches_in_file is None:
        len_ = len(np.load(y_train + str(i) + ".npy"))
        batches_in_file_new = len_ // BATCH_SIZE
    else:
        batches_in_file_new = batches_in_file

    if batches_in_file_new != 1:
        batch_index = i % batches_in_file_new
        x_curr = get_curr_x_y.x_curr[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
        y_curr = get_curr_x_y.y_curr[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
    else:
        x_curr = get_curr_x_y.x_curr[:BATCH_SIZE]
        y_curr = get_curr_x_y.x_curr[:BATCH_SIZE]

    return x_curr, y_curr


def get_curr_x_y2(x_train, y_train, batches_in_file, i):
    if batches_in_file is None:
        len_ = len(np.load(y_train + str(i) + ".npy"))
        batches_in_file_new = len_ // BATCH_SIZE
    else:
        batches_in_file_new = batches_in_file

    if batches_in_file_new != 1:
        batch_index = i % batches_in_file_new
        x_curr = np.load(x_train + str(i) + ".npy")[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
        y_curr = np.load(y_train + str(i) + ".npy")[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
    else:
        x_curr = np.load(x_train + str(i) + ".npy")[:BATCH_SIZE]
        y_curr = np.load(y_train + str(i) + ".npy")[:BATCH_SIZE]

    return x_curr, y_curr


def get_generator(x_train, y_train, batches_in_file=None, num_of_files=None):

    if isinstance(x_train, str):
        if batches_in_file is None and num_of_files is None:
            print("batches_in_file and num_of_files are None")
            return

        def generate_():
            i = -1
            while True:
                i += 1
                if num_of_files is None:
                    i %= (TRAIN_SET_SIZE // (BATCH_SIZE * batches_in_file))
                else:
                    i %= num_of_files
                x_curr, y_curr = get_curr_x_y(x_train, y_train, batches_in_file, i)
                yield (x_curr, y_curr)
    else:
        def generate_():
            batch_size = BATCH_SIZE
            i = -1*batch_size
            while True:
                i += batch_size
                i = i % len(x_train)
                yield (x_train[i:i+batch_size], y_train[i:i+batch_size])
    return generate_


def get_generator_multitask(x_train1, y_train1, x_train2, y_train2, batches_in_file=None, num_of_files=None):
    if isinstance(x_train1, str):
        def generate_():
            i = -1
            while True:
                i += 1
                if num_of_files is None:
                    i %= (TRAIN_SET_SIZE // (BATCH_SIZE * batches_in_file))
                else:
                    i %= num_of_files
                x_curr1, y_curr1 = get_curr_x_y(x_train1, y_train1, batches_in_file, i)
                x_curr2, y_curr2 = get_curr_x_y(x_train2, y_train2, batches_in_file, i)
                yield ({"input1": x_curr1, "input2": x_curr2}, {"pred1": y_curr1, "pred2": y_curr2})
    else:
        def generate_():
            batch_size = BATCH_SIZE
            i = -1*batch_size
            while True:
                i += batch_size
                i = i % len(x_train1)
                yield ({"input1": x_train1[i:i+batch_size], "input2": x_train2[i:i+batch_size]},
                       {"pred1": y_train1[i:i+batch_size], "pred2": y_train2[i:i+batch_size]})

    return generate_
