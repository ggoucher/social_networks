"""
Dialogue classifier. Uses a neural network to classify dialogue by speaker.
"""

import csv
import nltk
from nltk.stem.lancaster import LancasterStemmer

from training import start_training
from classification import classify


def get_raw_training_data(filename="dialogue_data.csv"):
    """
    Takes a csv input of lines and organizes into a dictionary with keys being actors, values being their lines.

    @param: filename - the name of the csv file
    @return: a dictionary from actor to lines
    """

    output_dict = {}

    with open(filename, newline='') as f:  # sorts through our csv file and organizes our text
        reader = csv.reader(f)
        for row in reader:
            if row[0] in output_dict:  # checking to see if the name is already in the dictionary
                output_dict[row[0]].append(row[1])
            else:
                output_dict[row[0]] = [row[1]]

    return output_dict


def pre_process_words(words, stemmer):
    """
    Takes a list of words, stems them, and returns the list of stemmed words with no repeats.

    @param: words - a list of words
    @param: stemmer - the Lancaster Stemmer to perform stemming
    @return: a list of stemmed words with no duplicates
    """

    stemmed_list = []
    undesired_chars = ["?", ".", "!", ","]

    for word in words:
        if word not in undesired_chars:
            stemmed_list.append(stemmer.stem(word))

    return list(set(stemmed_list))  # ensures we have no repetitions


def organize_raw_training_data(raw_training_data, stemmer):
    """
    Takes our raw training data, stems and organizes it.

    @param: raw_training_data - the raw training data
    @param: stemmer - the Lancaster Stemmer
    @return: word_stems, classes, documents (line, actor)
    """

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

    word_stems = pre_process_words(words, stemmer)  # stems the objects in our word list

    return word_stems, classes, documents


def create_training_data(word_stems, classes, documents, stemmer):
    """
    Generates training data based on whether or not words are present in each given sentence and which sentence format
    is followed.

    @param: word_stems - a list of the word_stems found in pre-processing
    @param: classes - a list of the different types/formats of sentence class identified in pre-processing
    @param: documents - a collection of all the different tuples consisting of (lines, actor) pairs
    @param: stemmer - the Lancaster Stemmer which is part of NLTK
    @return: training_data (list of bags for each sentence), output (list of sentences and classes within them)
    """

    training_data = []
    output = []

    for sentence in documents:
        bag = []
        class_id = []
        sentence_stemmed = []

        for word in sentence[0]:
            sentence_stemmed.append(stemmer.stem(word))

        for word in word_stems:
            bag.append(1 if word in sentence_stemmed else 0)

        for c in classes:
            class_id.append(1 if c == sentence[1] else 0)

        training_data.append(bag)
        output.append(class_id)

    return training_data, output


def main():
    stemmer = LancasterStemmer()

    raw_training_data = get_raw_training_data()  # organizes the script into characters
    word_stems, classes, documents = organize_raw_training_data(raw_training_data, stemmer)
    training_data, output = create_training_data(word_stems, classes, documents, stemmer)

    # Comment this out if you have already trained once and don't want to re-train
    start_training(word_stems, classes, training_data, output)

    # Classify new sentences
    classify(word_stems, classes, "will you look into the mirror?")
    classify(word_stems, classes, "mithril, as light as a feather, and as hard as dragon scales.")
    classify(word_stems, classes, "the thieves!")


if __name__ == "__main__":
    main()
