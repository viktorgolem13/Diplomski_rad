from sklearn import metrics
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from lightgbm import LGBMClassifier

import preprocessing


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


if __name__ == '__main__':
    main()
