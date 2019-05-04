from keras.layers import LSTM, Bidirectional, Embedding
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Activation, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2

import tensorflow as tf
import numpy as np
from sklearn import metrics

import preprocessing
# import hyperparameters
import bipolarDataset
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


def run(x_train, x_test, y_train_one_hot, y_test, model, fit_generator=False):
    if fit_generator:
        generate = preprocessing.get_generator(x_train, y_train_one_hot)
        model.fit_generator(generate(), steps_per_epoch=20, epochs=10)
    else:
        model.fit(x_train, y_train_one_hot, nb_epoch=8, batch_size=BATCH_SIZE)

    pred = model.predict_classes(x_test, 10)
    print(pred)
    print(y_test)
    print('acc: ', metrics.accuracy_score(pred, y_test))
    print('f1: ', metrics.f1_score(pred, y_test))


def ff():
    x_train, x_test, y_train, y_test = preprocessing.get_data()
    x_train, x_test = preprocessing.vectorize_data_tfidf(x_train, x_test)
    model = get_ff_model((x_train.shape[1], ))
    run(x_train, x_test, y_train, y_test, model)


def lstm_with_embedding_layer():
    x_train, x_test, y_train, y_test = preprocessing.get_data()
    y_train_one_hot = preprocessing.class_one_hot(y_train)
    embedding_matrix, word_index, tokenizer = preprocessing.get_embedding_matrix(x_train)
    x_train = preprocessing.vectorize_with_tokenizer(x_train, tokenizer)
    x_test = preprocessing.vectorize_with_tokenizer(x_test, tokenizer)
    model = get_lstm_model(True, word_index, embedding_matrix)
    run(x_train, x_test, y_train_one_hot, y_test, model)


def lstm():
    # x_train, x_test, y_train, y_test = preprocessing.get_data()
    x_train, x_test, y_train, y_test = bipolarDataset.get_bipolar_disorder_data()
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
    
    data_per_iteration = BATCH_SIZE * 10
    num_of_batches = TRAIN_SET_SIZE // data_per_iteration
    for i in range(num_of_batches):
        # x_train, y_train = preprocessing.get_data(start_index=i*data_per_iteration, end_index=(i+1)*data_per_iteration, test_size=0)
        x_train, y_train = bipolarDataset.get_bipolar_disorder_data(start_index=int(i * data_per_iteration / 2), 
                                                                    skiprows_start=int((i+1) * data_per_iteration / 2), 
                                                                    skiprows_end=int((i+1) * data_per_iteration / 2 + 10**7), 
                                                                    nrows=data_per_iteration, test_size=0)

        x_train = preprocessing.add_features_and_vectorize(x_train, vectorize_function, embedding_index)
        np.save("x_train" + str(i) + ".npy", x_train)
        y_train_one_hot = preprocessing.class_one_hot(y_train)
        np.save("y_train" + str(i) + ".npy", y_train_one_hot)
    
    x_test, y_test = bipolarDataset.get_bipolar_disorder_data(start_index=num_of_batches * data_per_iteration / 2, 
                                                              skiprows_start=(num_of_batches+1) * data_per_iteration / 2 + 250, 
                                                              skiprows_end=(num_of_batches+1) * data_per_iteration / 2 + 10**7 + 250, 
                                                              nrows=data_per_iteration, test_size=1)
    # x_test, y_test = preprocessing.get_data(start_index=0, end_index=0, test_size=500)

    x_test = preprocessing.add_features_and_vectorize(x_test, vectorize_function, embedding_index)
    model = get_lstm_model(use_embedding_layer=False, input_shape=(x_train.shape[1], x_train.shape[2]))
    run("x_train", x_test, "y_train", y_test, model, True)


if __name__ == '__main__':
    lstm_memory_efficient()
