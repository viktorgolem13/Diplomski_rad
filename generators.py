import numpy as np
from constants import *


def get_curr_x_y(x_train_filenames, y_train_filenames, batch_counter, file_counter):
    if batch_counter == 0 and file_counter == 0:
        get_curr_x_y.x_curr = np.load(x_train_filenames[file_counter])
        get_curr_x_y.y_curr = np.load(y_train_filenames[file_counter])
        get_curr_x_y.batches_in_file = len(get_curr_x_y.y_curr) // BATCH_SIZE

    elif batch_counter >= get_curr_x_y.batches_in_file:
        batch_counter = 0
        file_counter += 1
        file_counter %= len(x_train_filenames)
        # print(x_train_filenames)
        # print(file_counter)
        get_curr_x_y.x_curr = np.load(x_train_filenames[file_counter])
        get_curr_x_y.y_curr = np.load(y_train_filenames[file_counter])
        get_curr_x_y.batches_in_file = len(get_curr_x_y.y_curr) // BATCH_SIZE
        # print("x_curr ", get_curr_x_y.x_curr.shape)
        # print("y_curr ", get_curr_x_y.y_curr.shape)
        # print("batches_in_file ", get_curr_x_y.batches_in_file)

    x_curr = get_curr_x_y.x_curr[batch_counter * BATCH_SIZE:(batch_counter + 1) * BATCH_SIZE]
    y_curr = get_curr_x_y.y_curr[batch_counter * BATCH_SIZE:(batch_counter + 1) * BATCH_SIZE]

    return x_curr, y_curr, batch_counter, file_counter


def get_generator(x_train_filenames, y_train_filenames):
    def generate_():
        batch_counter = -1
        file_counter = 0
        while True:
            batch_counter += 1
            print()
            print("batch_counter: ", batch_counter)
            print("file_counter: ", file_counter)
            x_curr, y_curr, batch_counter, file_counter = get_curr_x_y(x_train_filenames, y_train_filenames,
                                                                       batch_counter, file_counter)
            yield (x_curr, y_curr)

    return generate_


def get_generator_multitask(x_train1_filenames, y_train1_filenames, x_train2_filenames, y_train2_filenames):
    def generate_():
        batch_counter = -1
        file_counter = 0
        while True:
            batch_counter += 1
            x_curr1, y_curr1, _, _ = get_curr_x_y(x_train1_filenames, y_train1_filenames, batch_counter, file_counter)
            x_curr2, y_curr2, batch_counter, file_counter = get_curr_x_y(x_train2_filenames, y_train2_filenames,
                                                                         batch_counter, file_counter)
            yield ({"input1": x_curr1, "input2": x_curr2}, {"pred1": y_curr1, "pred2": y_curr2})
    return generate_


def get_generator_multitask2(x_train_list, y_train1_one_hot_list):
    def generate_():
        batch_counter = -1
        file_counter = 0
        while True:
            batch_counter += 1
            input_dict = dict()
            output_dict = dict()
            for i in range(len(x_train_list)):
                if i != len(x_train_list) - 1:
                    x_curr, y_curr, _, _ = get_curr_x_y(x_train_list[i], y_train1_one_hot_list[i],
                                                        batch_counter, file_counter)
                else:
                    x_curr, y_curr, batch_counter, file_counter = get_curr_x_y(x_train_list[i],
                                                                               y_train1_one_hot_list[i],
                                                                               batch_counter, file_counter)
                    input_dict["input" + str(i)] = x_curr
                    output_dict["pred" + str(i)] = y_curr
            yield (input_dict, output_dict)
    return generate_
