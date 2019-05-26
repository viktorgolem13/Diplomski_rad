from keras.layers import LSTM, Bidirectional, Embedding
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import Adam

import numpy as np
from sklearn import metrics

import preprocessing
import generators
import load_data
import load_smhd_datasets
# import hyperparameters
from constants import *
from time import time


def get_ff_model(input_shape):
    model = Sequential()

    model.add(Dense(output_dim=1000, input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Dense(output_dim=2))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.01)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def get_lstm_model(use_embedding_layer, word_index=None, embedding_matrix=None, input_shape=None):
    model = Sequential()

    if use_embedding_layer:
        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        model.add(embedding_layer)
        model.add(Bidirectional(LSTM(units=300, dropout=0.2, recurrent_dropout=0.2)))
    else:
        model.add(Bidirectional(LSTM(units=300, dropout=0.2, recurrent_dropout=0.2), input_shape=input_shape))

    model.add(Activation('sigmoid'))

    model.add(Dense(output_dim=2))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.0075)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


# if fit_generator then x_train, y_train_one_hot are files where these data is stored insted of actual data
def run(x_train, x_test, y_train_one_hot, y_test, model, fit_generator=False, epochs=8, steps_per_epoch=20):
    if fit_generator:
        generate = generators.get_generator(x_train, y_train_one_hot)
        model.fit_generator(generate(), steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=0)
    else:
        model.fit(x_train, y_train_one_hot, nb_epoch=epochs, batch_size=BATCH_SIZE)

    pred = model.predict_classes(x_test, 10)
    print(pred)
    print(y_test)
    print('acc: ', metrics.accuracy_score(pred, y_test))
    print('f1: ', metrics.f1_score(pred, y_test))

    return metrics.accuracy_score(pred, y_test), metrics.f1_score(pred, y_test)


def ff():
    x_train, x_test, y_train, y_test = load_data.get_depression_data()
    x_train, x_test = preprocessing.vectorize_data_tfidf(x_train, x_test)
    model = get_ff_model((x_train.shape[1], ))
    run(x_train, x_test, y_train, y_test, model)


def lstm_with_embedding_layer():
    x_train, x_test, y_train, y_test = load_data.get_depression_data()
    y_train_one_hot = preprocessing.class_one_hot(y_train)
    embedding_matrix, word_index, tokenizer = preprocessing.get_embedding_matrix(x_train)
    x_train = preprocessing.vectorize_with_tokenizer(x_train, tokenizer)
    x_test = preprocessing.vectorize_with_tokenizer(x_test, tokenizer)
    model = get_lstm_model(True, word_index, embedding_matrix)
    run(x_train, x_test, y_train_one_hot, y_test, model)


def lstm():
    # x_train, x_test, y_train, y_test = load_data.get_depression_data()
    # x_train, x_test, y_train, y_test = load_data.get_bipolar_disorder_data()
    # x_train, y_train = load_data.get_rsdd_data(set_="train")
    # x_test, y_test = load_data.get_rsdd_data(end_index=5, set_="validation")
    x_train, y_train = load_data.get_smhd_data(set_="train")
    x_test, y_test = load_data.get_smhd_data(end_index=5, set_="validation")
    y_train_one_hot = preprocessing.class_one_hot(y_train)
    vectorize_function = preprocessing.vectorize_data_glove
    embedding_index = preprocessing.get_embeddings_index()
    print(x_train[0])
    x_train = preprocessing.add_features_and_vectorize(x_train, vectorize_function, embedding_index)
    x_test = preprocessing.add_features_and_vectorize(x_test, vectorize_function, embedding_index)
    model = get_lstm_model(use_embedding_layer=False, input_shape=(x_train.shape[1], x_train.shape[2]))
    run(x_train, x_test, y_train_one_hot, y_test, model)


def lstm_memory_efficient():

    vectorize_function = preprocessing.vectorize_data_glove
    embedding_index = preprocessing.get_embeddings_index()
    
    data_per_iteration = 5  # BATCH_SIZE * 10

    count = 0
    i = 0
    x_train_filenames = []
    y_train_filenames = []
    while count < TRAIN_SET_SIZE // BATCH_SIZE:
        i += 1
        # x_train, y_train = load_data.get_depression_data(start_index=i*data_per_iteration,
        #                                                  end_index=(i+1)*data_per_iteration, test_size=0)
        # x_train, y_train = load_data.get_bipolar_disorder_data(start_index=i * data_per_iteration // 2,
        #                                                        skiprows_start=(i+1) * data_per_iteration // 2,
        #                                                       skiprows_end=(i+1) * data_per_iteration // 2 + 10**7,
        #                                                        nrows=data_per_iteration, test_size=0)
        x_train, y_train = load_data.get_rsdd_data(start_index=i*data_per_iteration,
                                                   end_index=(i + 1) * data_per_iteration, set_="train")

        x_train = preprocessing.add_features_and_vectorize(x_train, vectorize_function, embedding_index)
        y_train_one_hot = preprocessing.class_one_hot(y_train, 2)
        print(x_train.shape)
        print(y_train_one_hot.shape)
        for j in range(len(x_train) // BATCH_SIZE):
            np.save("x_train" + str(count) + ".npy", x_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE])
            np.save("y_train" + str(count) + ".npy", y_train_one_hot[j * BATCH_SIZE:(j + 1) * BATCH_SIZE])
            x_train_filenames.append("x_train" + str(count) + ".npy")
            y_train_filenames.append("y_train" + str(count) + ".npy")
            count += 1

    # x_test, y_test = load_data.get_bipolar_disorder_data(start_index=num_of_batches * data_per_iteration // 2,
    #                                                     skiprows_start=(num_of_batches+1) * data_per_iteration // 2,
    #                                                     skiprows_end=(num_of_batches+1) * data_per_iteration // 2 + 10**7,
    #                                                     nrows=data_per_iteration, test_size=1)
    # x_test, y_test = load_data.get_depression_data(start_index=0, end_index=0, test_size=500)
    x_test, y_test = load_data.get_rsdd_data(end_index=5, set_="validation")

    x_test = preprocessing.add_features_and_vectorize(x_test, vectorize_function, embedding_index)
    np.save("x_test.npy", x_test)
    np.save("y_test.npy", y_test)

    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    model = get_lstm_model(use_embedding_layer=False, input_shape=(x_test.shape[1], x_test.shape[2]))
    return run(x_train_filenames, x_test, y_train_filenames, y_test, model, True, steps_per_epoch=20)


def save_data(vectorize_function, embedding_index, data_per_iteration=2, num_of_load_iterations=2):
    x_train_filenames = []
    y_train_filenames = []

    num_of_train_batches = 0
    for i in range(257, num_of_load_iterations):
        start = time()
        # x_train, y_train = load_data.get_rsdd_data(start_index=i * data_per_iteration,
        #                                            end_index=(i + 1) * data_per_iteration, set_="train")
        x0, x1, _, _, _ = load_smhd_datasets.get_smhd_data_user_level(start_index=i * data_per_iteration,
                                                                      end_index=(i + 1) * data_per_iteration)
        t1 = time()
        print(t1 - start)
        x_train, y_train = load_smhd_datasets.prepare_binary_data(x0, x1)
        t2 = time()
        print(t2 - t1)
        x_train = preprocessing.add_features_and_vectorize(x_train, vectorize_function, embedding_index)
        t3 = time()
        print(t3 - t2)
        y_train_one_hot = preprocessing.class_one_hot(y_train, 2)
        # print(x_train.shape)
        # print(y_train_one_hot.shape)
        np.save("x_train" + str(i) + ".npy", x_train)
        np.save("y_train" + str(i) + ".npy", y_train_one_hot)
        x_train_filenames.append("x_train" + str(i) + ".npy")
        y_train_filenames.append("y_train" + str(i) + ".npy")
        num_of_train_batches += len(x_train) // BATCH_SIZE
        end = time()
        print(end - t3)
    f = open("num_of_train_batches.txt", "w")
    f.write(str(num_of_train_batches))
    f.close()

    return x_train_filenames, y_train_filenames, num_of_train_batches


def lstm_memory_efficient_simple(reload_data=True, data_per_iteration=2, num_of_load_iterations=2,
                                 num_of_train_batches=None):
    if reload_data:
        vectorize_function = preprocessing.vectorize_data_glove
        embedding_index = preprocessing.get_embeddings_index()
        x_train_filenames, y_train_filenames, num_of_train_batches = save_data(vectorize_function, embedding_index,
                                                                               data_per_iteration,
                                                                               num_of_load_iterations)
        x0, x1, _, _, _ = load_smhd_datasets.get_smhd_data_user_level(end_index=100, set_="validation")
        x_test, y_test = load_smhd_datasets.prepare_binary_data(x0, x1)

        x_test = preprocessing.add_features_and_vectorize(x_test, vectorize_function, embedding_index)
        np.save("x_test.npy", x_test)
        np.save("y_test.npy", y_test)
    else:
        x_train_filenames = []
        y_train_filenames = []
        for i in range(num_of_load_iterations):
            x_train_filenames.append("x_train" + str(i) + ".npy")
            y_train_filenames.append("y_train" + str(i) + ".npy")
        if isinstance(num_of_train_batches, str):
            num_of_train_batches = eval(open(num_of_train_batches, "r").readlines()[0])

    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    model = get_lstm_model(use_embedding_layer=False, input_shape=(x_test.shape[1], x_test.shape[2]))
    return run(x_train_filenames, x_test, y_train_filenames, y_test, model, fit_generator=True,
               epochs=5, steps_per_epoch=num_of_train_batches)


if __name__ == '__main__':
    lstm_memory_efficient_simple(reload_data=False, data_per_iteration=2, num_of_load_iterations=2,
                                 num_of_train_batches=113)
