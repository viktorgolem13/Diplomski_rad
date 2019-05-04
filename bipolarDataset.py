import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import shuffle
from constants import *


def get_bipolar_disorder_data(start_index = 0, skiprows_start=10**3, skiprows_end=10**7, nrows = 2 * 10**3, test_size=0.2):
    df = pd.read_csv(BIPOLAR_DATA_DIR + "bipolar_control_reddit.csv",
                     skiprows=list(range(1, start_index)) + list(range(skiprows_start, skiprows_end)), nrows=nrows)
    print(df.head())
    df = df[["body", "classification"]]
    df = df.dropna(axis=0, how='any')
    x = df["body"].values
    y = df["classification"].values

    x, y = shuffle(x, y)

    if test_size > 1:
        test_size = test_size / nrows
    
    if test_size == 0 or test_size == 1:
        return x, y

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_train_, x_test_, y_train_, y_test_ = get_bipolar_disorder_data()
    print(x_train_[0])
    print(x_train_.shape)
    print(x_test_.shape)
    print(y_train_.shape)
    print(y_test_.shape)
