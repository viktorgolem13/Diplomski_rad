import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics

from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from lightgbm import LGBMClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


def get_data():
    x_train_positive = open("train_positive.txt", "r", encoding="utf8").readlines()
    x_train_negative = open("train_negative.txt", "r", encoding="utf8").readlines()
    x_test_positive = open("test_positive.txt", "r", encoding="utf8").readlines()
    x_test_negative = open("test_negative.txt", "r", encoding="utf8").readlines()

    x_train_positive = np.asarray(x_train_positive, dtype='<U2')
    x_train_negative = np.asarray(x_train_negative, dtype='<U2')
    x_test_positive = np.asarray(x_test_positive, dtype='<U2')
    x_test_negative = np.asarray(x_test_negative, dtype='<U2')

    y_train_positive = np.ones_like(x_train_positive)
    y_train_negative = np.zeros_like(x_train_negative)
    y_test_positive = np.ones_like(x_test_positive)
    y_test_negative = np.zeros_like(x_test_negative)

    x_train = np.concatenate((x_train_positive, x_train_negative), axis=0)
    y_train = np.concatenate((y_train_positive, y_train_negative), axis=0)

    x_test = np.concatenate((x_test_positive, x_test_negative), axis=0)
    y_test = np.concatenate((y_test_positive, y_test_negative), axis=0)

    return x_train, y_train, x_test, y_test


def preprocess_data(x_train_str, x_test_str):
    sw = stopwords.words("english")
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=sw, binary=True, sublinear_tf=True, norm=None)

    x_train = vectorizer.fit_transform(x_train_str).toarray()
    x_test = vectorizer.transform(x_test_str).toarray()

    return x_train, x_test


def baseline(x_train, y_train, x_test, y_test):
    rf = ensemble.RandomForestClassifier(max_depth=8, min_samples_split=4, max_features=0.5, n_estimators=100)
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    test_acc = metrics.accuracy_score(pred, y_test)
    print('Random forest acc: ' + str(test_acc))
    return pred


def main():
    x_train, y_train, x_test, y_test = get_data()
    x_train, x_test = preprocess_data(x_train, x_test)
    baseline(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
