#!/usr/bin/env python

"""
LfD Final Assignment - Naive Bayes

Available command-line options:
-t Define the train file to use. Uses data/train.tsv defuault.
-d Define the dev / test file to use. Uses data/dev.tsv defuault
-p Displays a plotted version of the confusion matrix for the report. May not always work
                             from the commandline if display packages are not installed.

Baseline usage:
python classic_models.py

How to run best model:


"""

import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from collections import Counter


def create_arg_parser():
    """Creates argumentparser and defines command-line options that can be called upon."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default='data/train.tsv', type=str,
                        help="Train file (default: data/train.txt)")
    parser.add_argument("-d", "--dev_file", default='data/dev.tsv', type=str,
                        help="Dev/test file (default: data/dev.txt)")
    parser.add_argument("-p", "--plot_show", action="store_true",
                        help="Displays a plotted version of the confusion matrix for the report. May not always work"
                             "from the commandline if display packages are not installed.")

    return parser.parse_args()


def read_corpus(corpus_file):
    """Reads the corpus file and splits lines into lists of
    documents and corresponding lists of labels."""
    documents, labels = [], []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('\t')
            documents.append(tokens[0])
            labels.append(tokens[1])

    print('*** Label Distribution')
    print(Counter(labels))
    return documents, labels


def print_evaluation(Y_test, Y_pred):
    """Takes true labels and predicted labels and
    prints evaluation measures (a classification report
    and confusion matrix)"""
    print('\n*** CLASSIFICATION REPORT ***')
    print(classification_report(Y_test, Y_pred))

    labels = ['NOT', 'OFF']
    print('\n*** CONFUSION MATRIX ***')
    cm = confusion_matrix(Y_test, Y_pred, labels=labels)
    print(' '.join(labels))
    print(cm)

    # If the --plot_show argument is given, show pyplt version of the confusion matrix
    if args.plot_show:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=labels)
        disp.plot()
        plt.show()


def identity(inp):
    '''Dummy function that just returns the input'''
    return inp


def train_and_evaluate(vec, X_train, Y_train, X_test, Y_test):
    """Trains and evaluates the NB classifier"""
    classifier = Pipeline([('vec', vec), ('cls', MultinomialNB(alpha=0.5))])
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    print(f'***** Evaluation of MultinomialNB *****')
    print_evaluation(Y_test, Y_pred)


if __name__ == "__main__":
    # Create argument parser
    args = create_arg_parser()

    # Load data used for experiments
    X_train, Y_train = read_corpus(args.train_file)
    X_test, Y_test = read_corpus(args.dev_file)

    # Choose vectorizer and features with hyperparameters for tuning
    vec = CountVectorizer(preprocessor=identity, tokenizer=identity)

    train_and_evaluate(vec, X_train, Y_train, X_test, Y_test)