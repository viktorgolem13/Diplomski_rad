import numpy as np

import preprocessing
import multitask2
import load_smhd_datasets
from constants import *


def multitask_smhd_memory_efficient(reload_data=True, data_per_iteration=2, num_of_load_iterations=2,
                                    num_of_train_batches=None, user_level=True):
    x_train1_filenames = []
    y_train1_filenames = []
    x_train2_filenames = []
    y_train2_filenames = []
    if reload_data:
        vectorize_function = preprocessing.vectorize_data_glove
        embedding_index = preprocessing.get_embeddings_index()

        num_of_train_batches = 0
        for i in range(num_of_load_iterations):
            if user_level:
                x0, x1, x2, _, _ = load_smhd_datasets.get_smhd_data_user_level(start_index=i * data_per_iteration,
                                                                               end_index=(i + 1) * data_per_iteration)
            else:
                x0, x1, x2, _, _ = load_smhd_datasets.get_smhd_data(start_index=i * data_per_iteration,
                                                                    end_index=(i + 1) * data_per_iteration)

            x_train1, y_train1 = load_smhd_datasets.prepare_binary_data(x0[:len(x0)//2], x1)
            x_train2, y_train2 = load_smhd_datasets.prepare_binary_data(x0[len(x0)//2:], x2)

            x_train1 = preprocessing.add_features_and_vectorize(x_train1, vectorize_function, embedding_index)
            x_train2 = preprocessing.add_features_and_vectorize(x_train2, vectorize_function, embedding_index)

            np.save("x_train1" + str(i) + ".npy", x_train1)
            y_train_one_hot1 = preprocessing.class_one_hot(y_train1, 2)
            np.save("y_train1" + str(i) + ".npy", y_train_one_hot1)

            np.save("x_train2" + str(i) + ".npy", x_train2)
            y_train_one_hot2 = preprocessing.class_one_hot(y_train2, 2)
            np.save("y_train2" + str(i) + ".npy", y_train_one_hot2)
            x_train1_filenames.append("x_train1" + str(i) + ".npy")
            y_train1_filenames.append("y_train1" + str(i) + ".npy")
            x_train2_filenames.append("x_train2" + str(i) + ".npy")
            y_train2_filenames.append("y_train2" + str(i) + ".npy")
            num_of_train_batches += len(x_train1) // BATCH_SIZE

        f = open("num_of_train_batches.txt", "w")
        f.write(num_of_train_batches)
        f.close()
        if user_level:
            x0, x1, x2, _, _ = load_smhd_datasets.get_smhd_data_user_level(set_='validation')
        else:
            x0, x1, x2, _, _ = load_smhd_datasets.get_smhd_data(set_='validation')

        x_test1, y_test1 = load_smhd_datasets.prepare_binary_data(x0[:len(x0)//2], x1)
        x_test2, y_test2 = load_smhd_datasets.prepare_binary_data(x0[len(x0)//2:], x2)

        x_test1 = preprocessing.add_features_and_vectorize(x_test1, vectorize_function, embedding_index)
        x_test2 = preprocessing.add_features_and_vectorize(x_test2, vectorize_function, embedding_index)
        x_test1 = x_test1[:len(x_test2)]
        y_test1 = y_test1[:len(y_test2)]
        x_test2 = x_test2[:len(x_test1)]
        y_test2 = y_test2[:len(y_test1)]
        print(len(y_test1))
        print(len(y_test2))
        np.save("x_test1.npy", x_test1)
        np.save("y_test1.npy", y_test1)
        np.save("x_test2.npy", x_test2)
        np.save("y_test2.npy", y_test2)

    else:
        for i in range(num_of_load_iterations):
            x_train1_filenames.append("x_train1" + str(i) + ".npy")
            y_train1_filenames.append("y_train1" + str(i) + ".npy")
            x_train2_filenames.append("x_train2" + str(i) + ".npy")
            y_train2_filenames.append("y_train2" + str(i) + ".npy")
        if isinstance(num_of_train_batches, str):
            num_of_train_batches = eval(open(num_of_train_batches, "r").readlines()[0])

    x_test1 = np.load("x_test1.npy")
    y_test1 = np.load("y_test1.npy")
    x_test2 = np.load("x_test2.npy")
    y_test2 = np.load("y_test2.npy")

    model = multitask2.get_multitask_model((x_test1.shape[1], x_test1.shape[2]), num_of_tasks=2)
    multitask2.run_multitask([x_train1_filenames, x_train2_filenames], [x_test1, x_test2],
                             [y_train1_filenames, y_train2_filenames], [y_test1, y_test2], model, fit_generator=True,
                             steps_per_epoch=num_of_train_batches)


if __name__ == '__main__':
    multitask_smhd_memory_efficient()
