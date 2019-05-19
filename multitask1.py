from keras.layers import LSTM
from keras.layers import Dense, Input, concatenate, Embedding
from keras.models import Model
from keras.optimizers import Adam

from sklearn import metrics
import preprocessing
import generators
from constants import *


def get_multitask_model(input_shape):
    inputs1 = Input(shape=input_shape, name="input1")
    inputs2 = Input(shape=input_shape, name="input2")

    shared_lstm = LSTM(units=300, dropout=0.2, recurrent_dropout=0.2)

    lstm_output1 = shared_lstm(inputs1)
    lstm_output2 = shared_lstm(inputs2)

    output1 = Dense(output_dim=1000, activation='relu')(lstm_output1)
    output2 = Dense(output_dim=1000, activation='relu')(lstm_output2)

    pred1 = Dense(output_dim=2, activation='softmax', name="pred1")(output1)
    pred2 = Dense(output_dim=2, activation='softmax', name="pred2")(output2)

    model = Model(inputs=[inputs1, inputs2], outputs=[pred1, pred2])

    adam = Adam(lr=0.0075)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def get_multitask_model_2_lstm(input_shape):
    inputs1 = Input(shape=input_shape)
    inputs2 = Input(shape=input_shape)

    shared_lstm = LSTM(units=300, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)
    lstm_output1 = shared_lstm(inputs1)
    lstm_output2 = shared_lstm(inputs2)

    output1 = LSTM(units=300, dropout=0.2, recurrent_dropout=0.2)(lstm_output1)
    output2 = LSTM(units=300, dropout=0.2, recurrent_dropout=0.2)(lstm_output2)

    pred1 = Dense(output_dim=2, activation='softmax')(output1)
    pred2 = Dense(output_dim=2, activation='softmax')(output2)

    model = Model(inputs=[inputs1, inputs2], outputs=[pred1, pred2])

    adam = Adam(lr=0.0075)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def get_multitask_model_2_lstm_double_connected(input_shape):
    inputs1 = Input(shape=input_shape)
    inputs2 = Input(shape=input_shape)

    shared_lstm = LSTM(units=300, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)
    lstm_output1 = shared_lstm(inputs1)
    lstm_output2 = shared_lstm(inputs2)

    second_lstm_inputs1 = concatenate([lstm_output1, inputs1])
    second_lstm_inputs2 = concatenate([lstm_output2, inputs2])

    output1 = LSTM(units=300, dropout=0.2, recurrent_dropout=0.2)(second_lstm_inputs1)
    output2 = LSTM(units=300, dropout=0.2, recurrent_dropout=0.2)(second_lstm_inputs2)

    pred1 = Dense(output_dim=2, activation='softmax')(output1)
    pred2 = Dense(output_dim=2, activation='softmax')(output2)

    model = Model(inputs=[inputs1, inputs2], outputs=[pred1, pred2])

    adam = Adam(lr=0.0075)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def get_multitask_model_2_embeddings(input_shape, word_index, embedding_matrix):
    inputs1 = Input(shape=input_shape, name="input1")
    inputs2 = Input(shape=input_shape, name="input2")

    embedding_layer_shared = Embedding(len(word_index) + 1,
                                       EMBEDDING_DIM,
                                       weights=[embedding_matrix],
                                       input_length=MAX_SEQUENCE_LENGTH,
                                       trainable=True)

    embedding_layer1 = Embedding(len(word_index) + 1,
                                 EMBEDDING_DIM,
                                 weights=[embedding_matrix],
                                 input_length=MAX_SEQUENCE_LENGTH,
                                 trainable=True)

    embedding_layer2 = Embedding(len(word_index) + 1,
                                 EMBEDDING_DIM,
                                 weights=[embedding_matrix],
                                 input_length=MAX_SEQUENCE_LENGTH,
                                 trainable=True)

    embedding_1_shared = embedding_layer_shared(inputs1)
    embedding_2_shared = embedding_layer_shared(inputs2)
    embedding_1 = embedding_layer1(inputs1)
    embedding_2 = embedding_layer2(inputs2)

    lstm_input_1 = concatenate([embedding_1_shared, embedding_1])
    lstm_input_2 = concatenate([embedding_2_shared, embedding_2])

    lstm1 = LSTM(units=300, dropout=0.2, recurrent_dropout=0.2)
    lstm2 = LSTM(units=300, dropout=0.2, recurrent_dropout=0.2)

    lstm_output1 = lstm1(lstm_input_1)
    lstm_output2 = lstm2(lstm_input_2)

    output1 = Dense(output_dim=1000, activation='relu')(lstm_output1)
    output2 = Dense(output_dim=1000, activation='relu')(lstm_output2)

    pred1 = Dense(output_dim=2, activation='softmax')(output1)
    pred2 = Dense(output_dim=2, activation='softmax')(output2)

    model = Model(inputs=[inputs1, inputs2], outputs=[pred1, pred2])

    adam = Adam(lr=0.0075)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def run_multitask(x_train1, x_test1, y_train1_one_hot, y_test1, x_train2, x_test2, y_train2_one_hot, y_test2,
                  model, fit_generator=False, batches_in_file=None, num_of_files=None):
    if fit_generator:
        generate = generators.get_generator_multitask(x_train1, y_train1_one_hot, x_train2, y_train2_one_hot,
                                                      batches_in_file, num_of_files)
        model.fit_generator(generate(), steps_per_epoch=20, epochs=10)
    else:
        model.fit([x_train1, x_train2], [y_train1_one_hot, y_train2_one_hot], nb_epoch=8, batch_size=50)

    pred = model.predict([x_test1, x_test2])

    pred1, pred2 = pred

    pred1 = preprocessing.one_hot_to_class(pred1)
    pred2 = preprocessing.one_hot_to_class(pred2)

    print("dataset 1")
    print(pred1)
    print(y_test1)
    print('acc: ', metrics.accuracy_score(pred1, y_test1))
    print('f1: ', metrics.f1_score(pred1, y_test1))

    print("dataset 2")
    print(pred2)
    print(y_test2)
    print('acc: ', metrics.accuracy_score(pred2, y_test2))
    print('f1: ', metrics.f1_score(pred2, y_test2))
