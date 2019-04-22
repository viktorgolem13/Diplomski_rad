from keras.layers import LSTM, Bidirectional, Embedding
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Activation, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2

from sklearn import metrics

import preprocessing
import hyperparameters
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


def run(x_train, x_test, y_train, y_test, model):
    y_train_one_hot = preprocessing.class_one_hot(y_train)
    model.fit(x_train, y_train_one_hot, nb_epoch=16, batch_size=10)

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
    embedding_matrix, word_index, tokenizer = preprocessing.get_embedding_matrix(x_train)
    x_train, x_test = preprocessing.vectorize_with_tokenizer(x_train, x_test, tokenizer)
    model = get_lstm_model(True, word_index, embedding_matrix)
    run(x_train, x_test, y_train, y_test, model)


def lstm():
    x_train, x_test, y_train, y_test = preprocessing.get_data()
    vectorize_function = preprocessing.vectorize_data_glove
    embedding_index = preprocessing.get_embeddings_index()
    x_train = preprocessing.add_features_and_vectorize(x_train, vectorize_function, embedding_index)
    x_test = preprocessing.add_features_and_vectorize(x_test, vectorize_function, embedding_index)
    model = get_lstm_model(use_embedding_layer=False, input_shape=(x_train.shape[1], x_train.shape[2]))
    run(x_train, x_test, y_train, y_test, model)


if __name__ == '__main__':
    lstm()
