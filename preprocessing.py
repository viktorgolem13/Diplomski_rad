from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from constants import *
import additional_fetures as af


def vectorize_data_tfidf(x_str, vectorizer):
    return vectorizer.transform(x_str).toarray()


def get_tfidf_vectorizer(x_train_str):
    sw = stopwords.words("english")
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=sw, binary=True, sublinear_tf=True, norm=None)

    vectorizer.fit(x_train_str)

    return vectorizer


def get_embedding_matrix(x_train_str, return_tokenizer=True):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(x_train_str)

    word_index = tokenizer.word_index

    embeddings_index = {}
    f = open(GLOVE_DIR, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except ValueError:
            pass
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    if return_tokenizer:
        return embedding_matrix, word_index, tokenizer
    else:
        return embedding_matrix, word_index


def get_embeddings_index():
    embeddings_index = {}
    f = open(GLOVE_DIR, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except:
            pass
    f.close()
    return embeddings_index


def vectorize_with_tokenizer(x_str, tokenizer):
    x = tokenizer.texts_to_sequences(x_str)
    return pad_sequences(x, maxlen=MAX_SEQUENCE_LENGTH)


def vectorize_data_1d_glove(x_str, embeddings_index):
    x = []
    for podatak in x_str:
        suma = []
        br = 0
        for word in podatak.split():
            word = word.lower()
            if len(word) > 1 and word[-1] in '.!?*,':
                word = word[:-1]
            if word in embeddings_index:
                if len(suma) == 0:
                    for broj in embeddings_index[word]:
                        suma.append(broj)
                    suma = np.asarray(suma)
                else:
                    suma += embeddings_index[word]
                br += 1
        if len(suma) == 0:
            for broj in embeddings_index['.']:
                suma.append(broj)
            suma = np.asarray(suma)
            br += 1
        x.append(suma / br)
    x = np.asarray(x)
    return x


def vectorize_data_glove(x_str, embeddings_index):
    x = []
    max_seq_len = -1
    for podatak in x_str:
        temp = []
        for word in podatak.split():
            word = word.lower()
            if len(word) > 1 and word[-1] in '.!?*,':
                word = word[:-1]
            if word in embeddings_index:
                temp.append(embeddings_index[word])

        temp = np.reshape(temp, (-1, EMBEDDING_DIM))
        if temp.shape[0] < MAX_BRANCH_LENGTH:
            temp = np.concatenate((temp, np.zeros((MAX_BRANCH_LENGTH - temp.shape[0], temp.shape[1]))), axis=0)
        else:
            if temp.shape[0] > max_seq_len:
                max_seq_len = temp.shape[0]
            temp = temp[:MAX_BRANCH_LENGTH]

        x.append(temp)
    x = np.reshape(x, (-1, MAX_BRANCH_LENGTH, EMBEDDING_DIM))
    print(x.shape)
    print("MAX SEQ LEN IS: ", max_seq_len)

    return x


def shuffle(x, y):
    rng_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rng_state)
    np.random.shuffle(y)
    return x, y


def one_hot_to_class(y_oh):
    return np.argmax(y_oh, axis=1)


def class_one_hot(y, num_of_classes=None):
    if num_of_classes is None:
        y_oh = np.zeros((len(y), max(y) + 1))
    else:
        y_oh = np.zeros((len(y), num_of_classes))
    for i in range(len(y)):
        y_oh[i, y[i]] = 1

    return y_oh


def add_features_and_vectorize(x_str, vectorize_function, vectorize_function_arg):
    additional_features = af.get_additional_features(x_str)
    x = vectorize_function(x_str, vectorize_function_arg)
    if len(x.shape) == 2:
        x = np.concatenate((x, additional_features), axis=1)
    else:
        additional_features = np.tile(additional_features, (1, 1, MAX_BRANCH_LENGTH))
        additional_features = np.reshape(additional_features, (x.shape[0], MAX_BRANCH_LENGTH, -1))
        x = np.concatenate((x, additional_features), axis=2)
    return x
