import numpy as np
from constants import *
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def length(x_str):
    return np.asarray([len(xx) for xx in x_str])


def all_caps(x_str):
    count_list = []
    for text in x_str:
        count = 0
        for word in text.split():
            if word.isupper():
                count += 1
        count_list.append(count)
    return np.asarray(count_list)


def num_of_uppercase_letters(x_str):
    count_list = []
    for text in x_str:
        count = 0
        for char in text:
            if char.isupper():
                count += 1
        count_list.append(count)
    return np.asarray(count_list)


def num_of_word_occurences(x_str, word='I'):
    count_list = []
    for text in x_str:
        count = 0
        for word_ in text.split():
            if word_ == word:
                count += 1
        count_list.append(count)
    return np.asarray(count_list)


def word_occurred(x_str, word='I'):
    has_occurred_list = []
    for text in x_str:
        has_occured = 0
        if word.strip() in text:
            has_occured = 1
        has_occurred_list.append(has_occured)
    return np.asarray(has_occurred_list)


def sentence_end(x_str):
    sentence_end_list = []
    for text in x_str:
        if text[-1] == ".":
            sentence_end_list.append(0)
        elif text[-1] == "?":
            sentence_end_list.append(1)
        elif text[-1] == "!":
            sentence_end_list.append(2)
        elif text[-1] == " ":
            sentence_end_list.append(3)
        else:
            sentence_end_list.append(4)
    return np.asarray(sentence_end_list)


def count_absolute_words(x_str):
    count_list = []
    for text in x_str:
        count = 0
        for absolutist_word in ABSOLUTIST_WORDS:
            if absolutist_word.strip() in text:
                count += 1
        count_list.append(count)
    return np.asarray(count_list)


def pos_tags(x_str, which_tag_to_count):
    count_list = []
    for text in x_str:
        count = 0
        for w, t in nltk.pos_tag(nltk.word_tokenize(text)):
            if t.lower().startswith(which_tag_to_count):
                count += 1
        count_list.append(count)
    return np.asarray(count_list)


def sentiment(x_str):
    sentiment_list = []
    sentiment_analyzer = SentimentIntensityAnalyzer()
    for text in x_str:
        sentiment_ = sentiment_analyzer.polarity_scores(text)['compound']
        sentiment_list.append(sentiment_)
    return np.asarray(sentiment_list)


def get_additional_features(x_str):
    additional_features = list()
    additional_features.append(length(x_str))
    additional_features.append(all_caps(x_str))
    additional_features.append(num_of_uppercase_letters(x_str))
    additional_features.append(sentence_end(x_str))
    additional_features.append(count_absolute_words(x_str))

    additional_features.append(num_of_word_occurences(x_str, 'I'))
    additional_features.append(num_of_word_occurences(x_str, 'me'))
    additional_features.append(num_of_word_occurences(x_str, 'myself'))

    additional_features.append(pos_tags(x_str, 'nn'))
    additional_features.append(pos_tags(x_str, 'vb'))
    additional_features.append(sentiment(x_str))

    num_features = len(additional_features)
    additional_features = np.asarray(additional_features).reshape(-1, num_features)
    return additional_features
