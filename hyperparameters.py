from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn import ensemble
from sklearn import linear_model
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier

from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Activation, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import LSTM, Bidirectional

from random import random, randint

import preprocessing
from constants import *


# funkcija koja traži najbolje hiperparametre random forest
def rf_hyperparametars(x_train, y_train, n_estimators):
    algorithm_parameters = dict()

    algorithm_parameters['min_samples_split'] = [2, 4, 6]

    algorithm_parameters['max_features'] = [0.01, 0.05, 0.1, 0.5, None, 'auto']

    algorithm_parameters['max_depth'] = [4, 8, 16, None]

    algorithm_parameters['criterion'] = ['gini', "entropy"]

    model = ensemble.RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)

    grid = GridSearchCV(model, algorithm_parameters)
    grid.fit(x_train, y_train)

    print(grid.best_estimator_.max_depth)
    print(grid.best_estimator_.min_samples_split)
    print(grid.best_estimator_.max_features)
    print(grid.best_estimator_.criterion)

    classifier = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                                 min_samples_split=grid.best_estimator_.min_samples_split,
                                                 max_depth=grid.best_estimator_.max_depth,
                                                 max_features=grid.best_estimator_.max_features,
                                                 criterion=grid.best_estimator_.criterion)

    return classifier


# hiperparametri za logističku regresiju
def lr_hyperparametars(x_train, y_train):
    algorithm_parameters = dict()

    c_values = []
    for i in range(1, 15):
        c_values.append(0.001 * 2 ** i)

    algorithm_parameters['C'] = c_values

    model = linear_model.LogisticRegression()

    grid = GridSearchCV(model, algorithm_parameters)
    grid.fit(x_train, y_train)

    print(grid.best_estimator_.C)

    classifier = linear_model.LogisticRegression(C=grid.best_estimator_.C)

    return classifier


def linear_svc_hyperparametars(x_train, y_train):
    algorithm_parameters = dict()

    c_values = []
    for i in range(1, 15):
        c_values.append(0.0001 * 2 ** i)

    algorithm_parameters['C'] = c_values

    model = LinearSVC()

    grid = GridSearchCV(model, algorithm_parameters)
    grid.fit(x_train, y_train)

    print(grid.best_estimator_.C)

    classifier = LinearSVC(C=grid.best_estimator_.C)

    return classifier


# funkcija koja traži najbolje hiperparametre gradient boostinga
def gb_hyperparametars(x_train, y_train):
    algorithm_parameters = dict()  # dict hiperparametara koje ćemo testirati

    # popunjavanje algorithm_parameters
    num_estimators_values = []
    for i in range(3, 6):
        num_estimators_values.append(4 ** i)

    num_estimators_values.append(100)

    algorithm_parameters['n_estimators'] = num_estimators_values

    learning_rate_values = []
    for i in range(3):
        learning_rate_values.append(0.01 * 10 ** i)

    algorithm_parameters['learning_rate'] = learning_rate_values

    algorithm_parameters['max_depth'] = [2, 4, 6]

    model = ensemble.GradientBoostingClassifier()

    # traženje i ispis najboljih hiperparametara
    grid = GridSearchCV(model, algorithm_parameters)
    print('searching...')
    grid.fit(x_train, y_train)

    print(grid.best_estimator_.n_estimators)
    print(grid.best_estimator_.max_depth)
    print(grid.best_estimator_.learning_rate)

    classifier = ensemble.GradientBoostingClassifier(n_estimators=grid.best_estimator_.n_estimators,
                                                     max_depth=grid.best_estimator_.max_depth,
                                                     learning_rate=grid.best_estimator_.learning_rate)

    return classifier


# funkcija koja traži najbolje hiperparametre gradient boostinga
def light_gb_hyperparametars(x_train, y_train):
    algorithm_parameters = dict()  # dict hiperparametara koje ćemo testirati

    algorithm_parameters['n_estimators'] = [100, 400, 1000]

    algorithm_parameters['learning_rate'] = [0.1, 0.02, 0.005]

    algorithm_parameters['max_depth'] = [10, -1]

    algorithm_parameters['num_leaves'] = [31, 2 ** 10, 2 ** 20]

    model = LGBMClassifier()

    # traženje i ispis najboljih hiperparametara
    grid = GridSearchCV(model, algorithm_parameters)
    print('searching...')
    grid.fit(x_train, y_train)

    print(grid.best_estimator_.n_estimators)
    print(grid.best_estimator_.max_depth)
    print(grid.best_estimator_.learning_rate)

    classifier = LGBMClassifier(n_estimators=grid.best_estimator_.n_estimators,
                                max_depth=grid.best_estimator_.max_depth,
                                learning_rate=grid.best_estimator_.learning_rate)

    return classifier


def deep_model_hyperparameters(x, y):
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2)
    y_validation = preprocessing.one_hot_to_class(y_validation)
    best_acc = 0
    best_f1 = 0
    best_avg = 0
    best_model = None
    for nb_epoch in [10, 30, 50]:
        for broj_slojeva in [1, 2]:
            for lr in [0.05, 0.01, 0.002, 0.0004]:
                for broj_neurona_u_sloju in [16, 32, 64, 128]:

                    model = Sequential()

                    for i in range(broj_slojeva):
                        if i == 0:
                            model.add(Dense(output_dim=broj_neurona_u_sloju, input_shape=(x_train.shape[1],)))
                            model.add(Activation('relu'))
                        else:
                            model.add(Dense(output_dim=broj_neurona_u_sloju))
                            model.add(Activation('relu'))

                    model.add(Dense(output_dim=3))
                    model.add(Activation('softmax'))

                    adam = Adam(lr=lr)

                    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

                    model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=100, verbose=0)

                    pred = model.predict_classes(x_validation, 100)

                    acc_score = metrics.accuracy_score(pred, y_validation)
                    f1_score = metrics.f1_score(pred, y_validation, average='macro')
                    avg_score = (acc_score + f1_score) / 2

                    if avg_score > best_avg:
                        print('acc_score: ', acc_score)
                        print('f1_score: ', f1_score)
                        print('avg_score: ', avg_score)
                        print('broj_slojeva: ', broj_slojeva)
                        print('broj_neurona: ', broj_neurona_u_sloju)
                        print('lr: ', lr)
                        print('nb_epoch: ', nb_epoch)
                        print()
                        best_avg = avg_score
                        best_model = model
                        if acc_score > best_acc:
                            best_acc = acc_score

                        if f1_score > best_f1:
                            best_f1 = f1_score

                    elif acc_score > best_acc:
                        print('acc_score: ', acc_score)
                        print('f1_score: ', f1_score)
                        print('avg_score: ', avg_score)
                        print('broj_slojeva: ', broj_slojeva)
                        print('broj_neurona: ', broj_neurona_u_sloju)
                        print('lr: ', lr)
                        print('nb_epoch: ', nb_epoch)
                        print()
                        best_acc = acc_score

                        if f1_score > best_f1:
                            best_f1 = f1_score

                    if f1_score > best_f1:
                        print('acc_score: ', acc_score)
                        print('f1_score: ', f1_score)
                        print('avg_score: ', avg_score)
                        print('broj_slojeva: ', broj_slojeva)
                        print('broj_neurona: ', broj_neurona_u_sloju)
                        print('lr: ', lr)
                        print('nb_epoch: ', nb_epoch)
                        print()
                        best_f1 = f1_score

    return best_model


def deep_model_hyperparameters_random_search(x, y, n_iterations=100):
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2)
    y_validation = preprocessing.one_hot_to_class(y_validation)
    best_acc = 0
    best_f1 = 0
    best_avg = 0
    best_model = None

    for _ in range(n_iterations):

        if random() < 0.5:
            nb_epoch = randint(5, 100)
        else:
            nb_epoch = randint(5, 30)  # model will not lose to much time for large epoch number

        if random() < 0.5:
            broj_slojeva = randint(1, 4)
        else:
            broj_slojeva = 1  # model with one hidden learining layer will be tested more

        if random() < 0.5:
            lr = random() * 10 ** (randint(0, 3) - 4)
        else:
            lr = random() * 10 ** -3

        if random() < 0.5:
            broj_neurona_u_sloju = randint(10, 100)
        else:
            broj_neurona_u_sloju = 2 ** randint(2, 10)

        # dropout either between zero and 0.5 or None
        r = random()
        if r < 0.3:
            dropout_rate = r
        elif r < 0.6:
            dropout_rate = random() / 10
        elif r < 0.7:
            dropout_rate = random() / 100
        else:
            dropout_rate = None

        r = random()
        if r < 0.2:
            kernel_regularizer_rate = r
            kernel_regularizer = l2(kernel_regularizer_rate)
        elif r < 0.4:
            kernel_regularizer_rate = random() / 10
            kernel_regularizer = l2(kernel_regularizer_rate)
        elif r < 0.6:
            kernel_regularizer_rate = random() / 100
            kernel_regularizer = l2(kernel_regularizer_rate)
        else:
            kernel_regularizer_rate = None
            kernel_regularizer = None

        if random() < 0.5:
            last_kernel_regularizer_rate = None
            last_kernel_regularizer = None
        else:
            last_kernel_regularizer_rate = kernel_regularizer_rate
            last_kernel_regularizer = l2(last_kernel_regularizer_rate)

        model = Sequential()

        for i in range(broj_slojeva):
            if i == 0:
                model.add(Dense(output_dim=broj_neurona_u_sloju, kernel_regularizer=kernel_regularizer,
                                input_shape=(x_train.shape[1],)))
                model.add(Activation('relu'))
                if dropout_rate is None:
                    model.add(Dropout(rate=dropout_rate))
            else:
                model.add(Dense(output_dim=broj_neurona_u_sloju, kernel_regularizer=kernel_regularizer))
                model.add(Activation('relu'))
                if dropout_rate is None:
                    model.add(Dropout(rate=dropout_rate))

        model.add(Dense(output_dim=3, kernel_regularizer=last_kernel_regularizer))
        model.add(Activation('softmax'))

        adam = Adam(lr=lr)

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=100, verbose=0)

        pred = model.predict_classes(x_validation, 100)

        acc_score = metrics.accuracy_score(pred, y_validation)
        f1_score = metrics.f1_score(pred, y_validation, average='macro')
        avg_score = (acc_score + f1_score) / 2

        if avg_score > best_avg:
            print('acc_score: ', acc_score)
            print('f1_score: ', f1_score)
            print('avg_score: ', avg_score)
            print('broj_slojeva: ', broj_slojeva)
            print('broj_neurona: ', broj_neurona_u_sloju)
            print('lr: ', lr)
            print('nb_epoch: ', nb_epoch)
            print('kernel_regularizer_rate: ', kernel_regularizer_rate)
            print('last_kernel_regularizer_rate: ', last_kernel_regularizer_rate)
            print('dropout_rate: ', dropout_rate)
            print()
            best_avg = avg_score
            best_model = model
            if acc_score > best_acc:
                best_acc = acc_score

            if f1_score > best_f1:
                best_f1 = f1_score

        elif acc_score > best_acc:
            print('acc_score: ', acc_score)
            print('f1_score: ', f1_score)
            print('avg_score: ', avg_score)
            print('broj_slojeva: ', broj_slojeva)
            print('broj_neurona: ', broj_neurona_u_sloju)
            print('lr: ', lr)
            print('nb_epoch: ', nb_epoch)
            print('kernel_regularizer_rate: ', kernel_regularizer_rate)
            print('last_kernel_regularizer_rate: ', last_kernel_regularizer_rate)
            print('dropout_rate: ', dropout_rate)
            print()
            best_acc = acc_score

            if f1_score > best_f1:
                best_f1 = f1_score

        if f1_score > best_f1:
            print('acc_score: ', acc_score)
            print('f1_score: ', f1_score)
            print('avg_score: ', avg_score)
            print('broj_slojeva: ', broj_slojeva)
            print('broj_neurona: ', broj_neurona_u_sloju)
            print('lr: ', lr)
            print('nb_epoch: ', nb_epoch)
            print('kernel_regularizer_rate: ', kernel_regularizer_rate)
            print('last_kernel_regularizer_rate: ', last_kernel_regularizer_rate)
            print('dropout_rate: ', dropout_rate)
            print()
            best_f1 = f1_score

    return best_model


def lstm_model_hyperparameters_random_search(x, y, n_iterations=100):
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2)
    y_train = preprocessing.class_one_hot(y_train)
    best_acc = 0
    best_f1 = 0
    best_avg = 0
    best_model = None

    for _ in range(n_iterations):

        if random() < 0.5:
            nb_epoch = randint(5, 50)
        else:
            nb_epoch = randint(5, 30)  # model will not lose to much time for large epoch number

        if random() < 0.5:
            lr = random() * 10 ** (randint(0, 3) - 4)
        else:
            lr = random() * 10 ** -3

        r = random()
        if r < 0.2:
            broj_neurona_u_sloju = randint(100, 500)
        elif r < 0.4:
            broj_neurona_u_sloju = 2 ** randint(2, 10)
        else:
            broj_neurona_u_sloju = EMBEDDING_DIM

        r = random()
        if r < 0.3:
            dropout_rate = r
        elif r < 0.5:
            dropout_rate = random() / 10
        elif r < 0.7:
            dropout_rate = 0.2
        else:
            dropout_rate = 0

        r = random()
        if r < 0.2:
            recurrent_dropout = r
        elif r < 0.4:
            recurrent_dropout = random() / 10
        elif r < 0.6:
            recurrent_dropout = 0.2
        elif r < 0.8:
            recurrent_dropout = 0
        else:
            recurrent_dropout = dropout_rate

        if random() < 0.5:
            bidirectional = True
        else:
            bidirectional = False

        model = Sequential()

        if bidirectional:
            model.add(Bidirectional(LSTM(units=broj_neurona_u_sloju, dropout=dropout_rate,
                                         recurrent_dropout=recurrent_dropout),
                                    input_shape=(x_train.shape[1], x_train.shape[2])))
        else:
            model.add(LSTM(units=broj_neurona_u_sloju, dropout=dropout_rate, recurrent_dropout=recurrent_dropout,
                           input_shape=(x_train.shape[1], x_train.shape[2])))

        model.add(Activation('sigmoid'))

        model.add(Dense(output_dim=2))
        model.add(Activation('softmax'))

        adam = Adam(lr=lr)

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=100, verbose=0)

        pred = model.predict_classes(x_validation, 100)

        acc_score = metrics.accuracy_score(pred, y_validation)
        f1_score = metrics.f1_score(pred, y_validation, average='macro')
        avg_score = (acc_score + f1_score) / 2

        if avg_score > best_avg:
            print('acc_score: ', acc_score)
            print('f1_score: ', f1_score)
            print('avg_score: ', avg_score)
            print('broj_neurona: ', broj_neurona_u_sloju)
            print('lr: ', lr)
            print('nb_epoch: ', nb_epoch)
            print('recurrent_dropout: ', recurrent_dropout)
            print('bidirectional: ', bidirectional)
            print('dropout_rate: ', dropout_rate)
            print()
            best_avg = avg_score
            best_model = model
            if acc_score > best_acc:
                best_acc = acc_score

            if f1_score > best_f1:
                best_f1 = f1_score

        elif acc_score > best_acc:
            print('acc_score: ', acc_score)
            print('f1_score: ', f1_score)
            print('avg_score: ', avg_score)
            print('broj_neurona: ', broj_neurona_u_sloju)
            print('lr: ', lr)
            print('nb_epoch: ', nb_epoch)
            print('recurrent_dropout: ', recurrent_dropout)
            print('bidirectional: ', bidirectional)
            print('dropout_rate: ', dropout_rate)
            print()
            best_acc = acc_score

            if f1_score > best_f1:
                best_f1 = f1_score

        if f1_score > best_f1:
            print('acc_score: ', acc_score)
            print('f1_score: ', f1_score)
            print('avg_score: ', avg_score)
            print('broj_neurona: ', broj_neurona_u_sloju)
            print('lr: ', lr)
            print('nb_epoch: ', nb_epoch)
            print('recurrent_dropout: ', recurrent_dropout)
            print('bidirectional: ', bidirectional)
            print('dropout_rate: ', dropout_rate)
            print()
            best_f1 = f1_score

    return best_model
