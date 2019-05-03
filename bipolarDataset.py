import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import shuffle
from constants import *


def get_bipolar_disorder_data():
    df = pd.read_csv(BIPOLAR_DATA_DIR + "bipolar_control_reddit.csv",
                     skiprows=range(2 * 10 ** 2, 10 ** 7), nrows=4 * 10 ** 2)
    df = df[["body", "classification"]]
    df = df.dropna(axis=0, how='any')
    print(df.head())
    x = df["body"].values
    y = df["classification"].values

    x, y = shuffle(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_train_, x_test_, y_train_, y_test_ = get_bipolar_disorder_data()
    print(x_train_[0])
    print(x_train_.shape)
    print(x_test_.shape)
    print(y_train_.shape)
    print(y_test_.shape)
