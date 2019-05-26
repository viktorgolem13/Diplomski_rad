from keras.layers import LSTM
from keras.layers import Dense, Input, concatenate, Embedding
from keras.models import Model
from keras.optimizers import Adam

from sklearn import metrics
import preprocessing
import generators
from constants import *


def get_multitask_model(input_shape, num_of_tasks):
    inputs_list = []
    for i in range(num_of_tasks):
        inputs_list.append(Input(shape=input_shape, name="input" + str(i)))

    shared_lstm = LSTM(units=300, dropout=0.2, recurrent_dropout=0.2)

    lstm_output_list = []
    for i in range(num_of_tasks):
        lstm_output_list.append(shared_lstm(inputs_list[i]))

    output_list = []
    for i in range(num_of_tasks):
        output_list.append(Dense(output_dim=1000, activation='relu')(lstm_output_list[i]))

    pred_list = []
    for i in range(num_of_tasks):
        pred_list.append(Dense(output_dim=2, activation='softmax', name="pred" + str(i))(output_list[i]))

    model = Model(inputs=inputs_list, outputs=pred_list)

    adam = Adam(lr=0.0075)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def run_multitask(x_train_list, x_test_list, y_train1_one_hot_list, y_test_list, model,
                  fit_generator=False, steps_per_epoch=20, epochs=10):
    if fit_generator:
        generate = generators.get_generator_multitask2(x_train_list, y_train1_one_hot_list)
        model.fit_generator(generate(), steps_per_epoch=steps_per_epoch, epochs=epochs)
    else:
        model.fit(x_train_list, y_train1_one_hot_list, nb_epoch=epochs, batch_size=50)

    pred_list = model.predict(x_test_list)

    for i in range(len(pred_list)):
        pred = preprocessing.one_hot_to_class(pred_list[i])

        print("dataset 1")
        print(pred)
        print(y_test_list[i])
        print('acc: ', metrics.accuracy_score(pred, y_test_list[i]))
        print('f1: ', metrics.f1_score(pred, y_test_list[i]))
