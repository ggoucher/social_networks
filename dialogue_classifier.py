import nltk
import numpy
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import csv
import numpy as np
import time
nltk.download('punkt')


def get_raw_training_data(input):
    """takes a csv input of lines and organizes into a dictionary with keys being actors, values being their lines"""

    output_dict = {}

    with open(input, newline='') as f:  # sorts through our csv file and organizes our text
        reader = csv.reader(f)
        for row in reader:
            if row[0] in output_dict:  # checking to see if the name is already in the dictionary
                output_dict[row[0]].append(row[1])
            else:
                output_dict[row[0]] = [row[1]]

    return output_dict


def preprocess_words(words, stemmer):
    """takes a list of words, stems them, and returns the list of stemmed words with no repeats"""

    stemmed_list = []

    undesired_chars = ["?", ".", "!", ","]

    for word in words:
        if word not in undesired_chars:
            stemmed_list.append(stemmer.stem(word))

    return list(set(stemmed_list))  # ensures we have no repetitions


def organize_raw_training_data(raw_training_data, stemmer):
    """takes our raw training data, stems and organizes it"""

    words = []  # list of all words
    documents = []  # list of all tuple classes
    classes = []  # our list of classes

    for key in raw_training_data:
        classes.append(key)  # makes a list of our actors

        char_words = []
        for value in raw_training_data[key]:
            tokenized_words = nltk.word_tokenize(value)
            words += tokenized_words
            char_words += tokenized_words
        documents.append((char_words, key))

    word_stems = preprocess_words(words, stemmer)  # stems the objects in our word list

    return word_stems, classes, documents


def create_training_data(word_stems, classes, documents, stemmer):
    """
    Generates training data based on whether or not words are present in each given sentence and which sentence format
    is followed.

    @param: word_stems A list of the word_stems found in preprocessing.
    @param: Classes A list of the different types/formats of sentence class identified in preprocessing.
    @param: Documents A collection of all the different tuples consisting of (lines, actor) pairs.
    @param: Stemmer The Lancaster Stemmer which is part of NLTK
    """

    training_data = []  # A list of bags of words for each sentence as represented in our docs.
    output = []  # A list of all the sentences and the classes within them.

    for sentence in documents:

        bag = []
        class_id = []
        sentence_stemmed = []

        for word in sentence[0]:

            sentence_stemmed.append(stemmer(word))

        for word in word_stems:

            if word in sentence_stemmed:

                bag.append(1)

            else:

                bag.append(0)

        for c in classes:

            if c is sentence[1]:

                class_id.append(1)

            else:

                class_id.append(0)

        training_data.append(bag)
        output.append(class_id)

    return training_data, output

def sigmoid(z):
    """ Takes in z and returns the basic sigmoid function for it. """
    return 1/((1) + numpy.exp(-z))

def sigmoid_output_to_derivative(output):
    """ Convert the sigmoid function's output to its derivative. """
    return output * (1 - output)


def main():
    stemmer = LancasterStemmer()

    raw_training_data = get_raw_training_data('dialogue_data.csv')  # organizes the script into characters
    word_stems, classes, documents = organize_raw_training_data(raw_training_data, stemmer)
    training_data, output = create_training_data(word_stems, classes, documents, stemmer)

if __name__ == "__main__":
    main()
