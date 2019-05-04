import tensorflow as tf
from keras.layers import LSTM
from keras.layers import Dense, Input, concatenate, Embedding
from keras.models import Model
from keras.optimizers import Adam

from sklearn import metrics

import preprocessing
import bipolarDataset
from constants import *


def get_multitask_model(input_shape):
    inputs1 = Input(shape=input_shape)
    inputs2 = Input(shape=input_shape)

    shared_lstm = LSTM(units=300, dropout=0.2, recurrent_dropout=0.2)

    lstm_output1 = shared_lstm(inputs1)
    lstm_output2 = shared_lstm(inputs2)

    output1 = Dense(output_dim=1000, activation='relu')(lstm_output1)
    output2 = Dense(output_dim=1000, activation='relu')(lstm_output2)

    pred1 = Dense(output_dim=2, activation='softmax')(output1)
    pred2 = Dense(output_dim=2, activation='softmax')(output2)

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
    inputs1 = Input(shape=input_shape)
    inputs2 = Input(shape=input_shape)

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


def run_multitask(x_train1, x_test1, y_train1, y_test1, x_train2, x_test2, y_train2, y_test2, model):
    y_train1_one_hot = preprocessing.class_one_hot(y_train1)
    y_train2_one_hot = preprocessing.class_one_hot(y_train2)
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


def multitask():
    x_train1, x_test1, y_train1, y_test1 = preprocessing.get_data()
    x_train2, x_test2, y_train2, y_test2 = bipolarDataset.get_bipolar_disorder_data()
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

    model = get_multitask_model((x_train1.shape[1], x_train1.shape[2]))

    run_multitask(x_train1, x_test1, y_train1, y_test1, x_train2, x_test2, y_train2, y_test2, model)


def multitask2():
    x_train1, x_test1, y_train1, y_test1 = preprocessing.get_data()
    x_train2, x_test2, y_train2, y_test2 = bipolarDataset.get_bipolar_disorder_data()
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

    model = get_multitask_model_2_embeddings((x_train1.shape[1], ), word_index, embedding_matrix)

    run_multitask(x_train1, x_test1, y_train1, y_test1, x_train2, x_test2, y_train2, y_test2, model)


def multitask_memory_efficient():
    vectorize_function = preprocessing.vectorize_data_glove
    embedding_index = preprocessing.get_embeddings_index()
    
    data_per_iteration = BATCH_SIZE
    num_of_batches = TRAIN_SET_SIZE // data_per_iteration
    for i in range(num_of_batches):
        x_train1, y_train1 = preprocessing.get_data(start_index=i*data_per_iteration, end_index=(i+1)*data_per_iteration, test_size=0)
        x_train2, y_train2 = bipolarDataset.get_bipolar_disorder_data(skiprows_start=(i+1) * data_per_iteration / 2, 
            skiprows_end=(i+1) * data_per_iteration / 2 + 10**7, nrows=data_per_iteration)

        x_train1 = preprocessing.add_features_and_vectorize(x_train1, vectorize_function, embedding_index)
        x_train2 = preprocessing.add_features_and_vectorize(x_train2, vectorize_function, embedding_index)

        x_train1 = x_train1[:len(x_train2)]
        x_test1 = x_test1[:len(x_test2)]
        y_train1 = y_train1[:len(y_train2)]
        y_test1 = y_test1[:len(y_test2)]

        np.save("x_train1" + str(i) + ".npy", x_train1)
        y_train_one_hot1 = preprocessing.class_one_hot(y_train1)
        np.save("y_train1" + str(i) + ".npy", y_train_one_hot1)

        np.save("x_train2" + str(i) + ".npy", x_train2)
        y_train_one_hot2 = preprocessing.class_one_hot(y_train2)
        np.save("y_train2" + str(i) + ".npy", y_train_one_hot2)
    
    x_test1, y_test1 = preprocessing.get_data(start_index=0, end_index=0, test_size=500)

    x_test2, y_test2 = bipolarDataset.get_bipolar_disorder_data(start_index=num_of_batches * data_per_iteration / 2, 
                                                              skiprows_start=(num_of_batches+1) * data_per_iteration / 2 + 250, 
                                                              skiprows_end=(num_of_batches+1) * data_per_iteration / 2 + 10**7 + 250, 
                                                              nrows=data_per_iteration, test_size=1)

    x_test1 = preprocessing.add_features_and_vectorize(x_test1, vectorize_function, embedding_index)
    x_test2 = preprocessing.add_features_and_vectorize(x_test2, vectorize_function, embedding_index)

    model = get_multitask_model((x_test1.shape[1], x_test1.shape[2]))
    run_multitask("x_train1", x_test1, "y_train1", y_test1, "x_train2", x_test2, "y_train2", y_test2, model)


if __name__ == '__main__':
    multitask2()
