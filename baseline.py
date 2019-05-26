from sklearn import metrics
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
# from lightgbm import LGBMClassifier

import preprocessing
import load_data
from constants import *
import load_smhd_datasets


def baseline(x_train, x_test, y_train, y_test):
    rf = ensemble.RandomForestClassifier(max_depth=8, min_samples_split=4, max_features=0.5, n_estimators=100)
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    test_acc = metrics.accuracy_score(pred, y_test)
    f1_score = metrics.accuracy_score(pred, y_test)
    print('Random forest acc: ', test_acc)
    print('Random forest f1 score: ', f1_score)
    return test_acc, f1_score


def main(train_set_size=1000):
    # x_train, x_test, y_train, y_test = load_data.get_depression_data(start_index=0, end_index=train_set_size//2,
    #                                                                   test_size=500)
    x0, x1, _, _, _ = load_smhd_datasets.get_smhd_data_user_level(start_index=0, end_index=train_set_size)
    x_train, y_train = load_smhd_datasets.prepare_binary_data(x0, x1)

    x0, x1, _, _, _ = load_smhd_datasets.get_smhd_data_user_level(end_index=100, set_="validation")
    x_test, y_test = load_smhd_datasets.prepare_binary_data(x0, x1)

    print(x_train.shape)
    vectorize_function = preprocessing.vectorize_data_tfidf
    vectorizer = preprocessing.get_tfidf_vectorizer(x_train)
    # x_train = preprocessing.add_features_and_vectorize(x_train, vectorize_function, vectorizer)
    # x_test = preprocessing.add_features_and_vectorize(x_test, vectorize_function, vectorizer)
    x_train = vectorize_function(x_train, vectorizer)
    x_test = vectorize_function(x_test, vectorizer)
    test_acc, f1_score = baseline(x_train, x_test, y_train, y_test)
    return test_acc, f1_score


def rf_with_glove():
    x_train, x_test, y_train, y_test = load_data.get_depression_data()
    vectorize_function = preprocessing.vectorize_data_1d_glove
    embedding_index = preprocessing.get_embeddings_index()
    x_train = preprocessing.add_features_and_vectorize(x_train, vectorize_function, embedding_index)
    x_test = preprocessing.add_features_and_vectorize(x_test, vectorize_function, embedding_index)
    pred = baseline(x_train, x_test, y_train, y_test)
    print(pred)


if __name__ == '__main__':
    main()
