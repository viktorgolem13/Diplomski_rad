from keras.layers import LSTM, Bidirectional, Embedding
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Activation, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2

import numpy as np
from sklearn import metrics

import preprocessing
import generators
import load_data
# import hyperparameters
from constants import *


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


def run(x_train, x_test, y_train_one_hot, y_test, model, fit_generator=False, batches_in_file=None, num_of_files=None):
    if fit_generator:
        generate = generators.get_generator(x_train, y_train_one_hot, batches_in_file, num_of_files)
        model.fit_generator(generate(), steps_per_epoch=20, epochs=10)
    else:
        model.fit(x_train, y_train_one_hot, nb_epoch=8, batch_size=BATCH_SIZE)

    pred = model.predict_classes(x_test, 10)
    print(pred)
    print(y_test)
    print('acc: ', metrics.accuracy_score(pred, y_test))
    print('f1: ', metrics.f1_score(pred, y_test))


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
    num_of_batches = 5  # TRAIN_SET_SIZE // data_per_iteration

    count = 0
    # for i in range(num_of_batches):
    i = 0
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
    run("x_train", x_test, "y_train", y_test, model, True)


if __name__ == '__main__':
    lstm()
