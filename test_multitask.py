import numpy as np

import load_data
import preprocessing
import multitask1
import load_smhd_datasets
from constants import *


def multitask():
    x_train1, x_test1, y_train1, y_test1 = load_data.get_depression_data()
    x_train2, x_test2, y_train2, y_test2 = load_data.get_bipolar_disorder_data()
    x_train1 = x_train1[:len(x_train2)]
    x_test1 = x_test1[:len(x_test2)]
    y_train1 = y_train1[:len(y_train2)]
    y_test1 = y_test1[:len(y_test2)]
    vectorize_function = preprocessing.vectorize_data_glove
    embedding_index = preprocessing.get_embeddings_index()

    x_train1 = preprocessing.add_features_and_vectorize(x_train1, vectorize_function, embedding_index)
    x_test1 = preprocessing.add_features_and_vectorize(x_test1, vectorize_function, embedding_index)
    x_train2 = preprocessing.add_features_and_vectorize(x_train2, vectorize_function, embedding_index)
    x_test2 = preprocessing.add_features_and_vectorize(x_test2, vectorize_function, embedding_index)

    model = multitask1.get_multitask_model((x_train1.shape[1], x_train1.shape[2]))

    multitask1.run_multitask(x_train1, x_test1, y_train1, y_test1, x_train2, x_test2, y_train2, y_test2, model)


def multitask2():
    x_train1, x_test1, y_train1, y_test1 = load_data.get_depression_data()
    x_train2, x_test2, y_train2, y_test2 = load_data.get_bipolar_disorder_data()
    x_train1 = x_train1[:len(x_train2)]
    x_test1 = x_test1[:len(x_test2)]
    y_train1 = y_train1[:len(y_train2)]
    y_test1 = y_test1[:len(y_test2)]
    vectorize_function = preprocessing.vectorize_with_tokenizer
    embedding_matrix, word_index, tokenizer = preprocessing.get_embedding_matrix(x_train1)

    x_train1 = vectorize_function(x_train1, tokenizer)
    x_train2 = vectorize_function(x_train2, tokenizer)
    x_test1 = vectorize_function(x_test1, tokenizer)
    x_test2 = vectorize_function(x_test2, tokenizer)

    model = multitask1.get_multitask_model_2_embeddings((x_train1.shape[1],), word_index, embedding_matrix)

    multitask1.run_multitask(x_train1, x_test1, y_train1, y_test1, x_train2, x_test2, y_train2, y_test2, model)


def multitask_memory_efficient():
    vectorize_function = preprocessing.vectorize_data_glove
    embedding_index = preprocessing.get_embeddings_index()

    data_per_iteration = BATCH_SIZE
    num_of_batches = TRAIN_SET_SIZE // data_per_iteration
    for i in range(num_of_batches):
        x_train1, y_train1 = load_data.get_depression_data(start_index=i * data_per_iteration,
                                                           end_index=(i + 1) * data_per_iteration, test_size=0)
        x_train2, y_train2 = load_data.get_bipolar_disorder_data(start_index=i * data_per_iteration // 2,
                                                                 skiprows_start=(i + 1) * data_per_iteration // 2,
                                                                 skiprows_end=(i + 1) * data_per_iteration // 2 + 10**7,
                                                                 nrows=data_per_iteration, test_size=0)

        x_train1 = preprocessing.add_features_and_vectorize(x_train1, vectorize_function, embedding_index)
        x_train2 = preprocessing.add_features_and_vectorize(x_train2, vectorize_function, embedding_index)

        x_train1 = x_train1[:len(x_train2)]
        y_train1 = y_train1[:len(y_train2)]

        np.save("x_train1" + str(i) + ".npy", x_train1)
        y_train_one_hot1 = preprocessing.class_one_hot(y_train1, 2)
        np.save("y_train1" + str(i) + ".npy", y_train_one_hot1)

        np.save("x_train2" + str(i) + ".npy", x_train2)
        y_train_one_hot2 = preprocessing.class_one_hot(y_train2, 2)
        np.save("y_train2" + str(i) + ".npy", y_train_one_hot2)

    x_test1, y_test1 = load_data.get_depression_data(start_index=0, end_index=0, test_size=500)

    x_test2, y_test2 = load_data.get_bipolar_disorder_data(start_index=num_of_batches * data_per_iteration // 2,
                                                           skiprows_start=(num_of_batches + 1) *
                                                           data_per_iteration // 2 + 250,
                                                           skiprows_end=(num_of_batches + 1) *
                                                           data_per_iteration // 2 + 10**7 + 250,
                                                           nrows=data_per_iteration, test_size=1)

    x_test1 = preprocessing.add_features_and_vectorize(x_test1, vectorize_function, embedding_index)
    x_test2 = preprocessing.add_features_and_vectorize(x_test2, vectorize_function, embedding_index)
    x_test1 = x_test1[:len(x_test2)]
    y_test1 = y_test1[:len(y_test2)]

    model = multitask1.get_multitask_model((x_test1.shape[1], x_test1.shape[2]))
    multitask1.run_multitask("x_train1", x_test1, "y_train1", y_test1, "x_train2", x_test2,
                             "y_train2", y_test2, model, fit_generator=True, batches_in_file=1)


def multitask_smhd():
    x0, _, x2, x1, _ = load_smhd_datasets.get_smhd_data()
    x_train1, y_train1 = load_smhd_datasets.prepare_binary_data(x0[:len(x0)//2], x1)
    x_train2, y_train2 = load_smhd_datasets.prepare_binary_data(x0[len(x0)//2:], x2)
    print(y_train1)
    x0, _, x2, x1, _ = load_smhd_datasets.get_smhd_data(set_='validation')
    x_test1, y_test1 = load_smhd_datasets.prepare_binary_data(x0[:len(x0)//2], x1)
    x_test2, y_test2 = load_smhd_datasets.prepare_binary_data(x0[len(x0)//2:], x2)
    print(y_test2)
    x_train1 = x_train1[:len(x_train2)]
    x_test1 = x_test1[:len(x_test2)]
    y_train1 = y_train1[:len(y_train2)]
    y_test1 = y_test1[:len(y_test2)]

    x_train2 = x_train2[:len(x_train1)]
    x_test2 = x_test2[:len(x_test1)]
    y_train2 = y_train2[:len(y_train1)]
    y_test2 = y_test2[:len(y_test1)]

    vectorize_function = preprocessing.vectorize_data_glove
    embedding_index = preprocessing.get_embeddings_index()

    x_train1 = preprocessing.add_features_and_vectorize(x_train1, vectorize_function, embedding_index)
    x_test1 = preprocessing.add_features_and_vectorize(x_test1, vectorize_function, embedding_index)
    x_train2 = preprocessing.add_features_and_vectorize(x_train2, vectorize_function, embedding_index)
    x_test2 = preprocessing.add_features_and_vectorize(x_test2, vectorize_function, embedding_index)

    model = multitask1.get_multitask_model((x_test1.shape[1], x_test1.shape[2]))
    print(y_train1)
    multitask1.run_multitask(x_train1, x_test1, y_train1, y_test1, x_train2, x_test2, y_train2, y_test2, model)


def multitask_smhd_memory_efficient():
    vectorize_function = preprocessing.vectorize_data_glove
    embedding_index = preprocessing.get_embeddings_index()

    data_per_iteration = 50
    num_of_load_iterations = 2
    for i in range(num_of_load_iterations):
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

    model = multitask1.get_multitask_model((x_test1.shape[1], x_test1.shape[2]))
    multitask1.run_multitask("x_train1", x_test1, "y_train1", y_test1, "x_train2", x_test2,
                             "y_train2", y_test2, model, fit_generator=True, num_of_files=num_of_load_iterations)


if __name__ == '__main__':
    multitask_smhd_memory_efficient()
