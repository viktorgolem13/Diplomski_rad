from sklearn import metrics
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
# from lightgbm import LGBMClassifier

import preprocessing
from constants import *


def baseline(x_train, x_test, y_train, y_test):
    rf = ensemble.RandomForestClassifier(max_depth=8, min_samples_split=4, max_features=0.5, n_estimators=100)
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    test_acc = metrics.accuracy_score(pred, y_test)
    print('Random forest acc: ', test_acc)
    return pred


def main():
    x_train, x_test, y_train, y_test = preprocessing.get_data()
    print(x_train.shape)
    vectorize_function = preprocessing.vectorize_data_tfidf
    vectorizer = preprocessing.get_tfidf_vectorizer(x_train)
    x_train = preprocessing.add_features_and_vectorize(x_train, vectorize_function, vectorizer)
    x_test = preprocessing.add_features_and_vectorize(x_test, vectorize_function, vectorizer)
    pred = baseline(x_train, x_test, y_train, y_test)
    print(pred)


def rf_with_glove():
    x_train, x_test, y_train, y_test = preprocessing.get_data()
    vectorize_function = preprocessing.vectorize_data_1d_glove
    embedding_index = preprocessing.get_embeddings_index()
    x_train = preprocessing.add_features_and_vectorize(x_train, vectorize_function, embedding_index)
    x_test = preprocessing.add_features_and_vectorize(x_test, vectorize_function, embedding_index)
    pred = baseline(x_train, x_test, y_train, y_test)
    print(pred)


def main_memory_efficient():
    vectorize_function = preprocessing.vectorize_data_tfidf
    vectorizer = preprocessing.get_tfidf_vectorizer(x_train)
    
    data_per_iteration = BATCH_SIZE
    num_of_batches = TRAIN_SET_SIZE // data_per_iteration
    for i in range(num_of_batches):
        # x_train, y_train = preprocessing.get_data(start_index=i*data_per_iteration, end_index=(i+1)*data_per_iteration, test_size=0)
        x_train, y_train = bipolarDataset.get_bipolar_disorder_data(start_index=int(i * data_per_iteration / 2), skiprows_start=int((i+1) * data_per_iteration / 2), 
            skiprows_end=int((i+1) * data_per_iteration / 2 + 10**7), nrows=data_per_iteration, test_size=0)

        x_train = preprocessing.add_features_and_vectorize(x_train, vectorize_function, vectorizer)
        np.save("x_train" + str(i) + ".npy", x_train)
        y_train_one_hot = preprocessing.class_one_hot(y_train)
        np.save("y_train" + str(i) + ".npy", y_train_one_hot)
    
    x_test, y_test = bipolarDataset.get_bipolar_disorder_data(start_index=num_of_batches * data_per_iteration / 2, 
                                                              skiprows_start=(num_of_batches+1) * data_per_iteration / 2 + 250, 
                                                              skiprows_end=(num_of_batches+1) * data_per_iteration / 2 + 10**7 + 250, 
                                                              nrows=data_per_iteration, test_size=1)

    x_test = preprocessing.add_features_and_vectorize(x_test, vectorize_function, vectorizer)

    pred = baseline(x_train, x_test, y_train, y_test)
    print(pred)


if __name__ == '__main__':
    main()
