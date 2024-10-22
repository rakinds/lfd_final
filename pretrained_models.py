#!/usr/bin/env python

"""
LfD Final Assignment - Pretrained Models

This script allows the user to run several Pretrained Models on a binary classification task.

Available command-line options:
-i  Input file to learn from (default data/train.tsv)
-d  The dev set to read in (default data/dev.tsv)")
-t  The test set to read in (default data/test.tsv)"

How to run the best configuration:


"""

import argparse
import random
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Make reproducible
np.random.seed(1234)
tf.random.set_seed(1234)
random.seed(1234)


def create_arg_parser():
    """Create an ArgumentParser to parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='data/train.tsv', type=str,
                        help="Input file to learn from (default data/train.tsv)")
    parser.add_argument("-d", "--dev_file", type=str, default='data/dev.tsv',
                        help="The dev set to read in (default data/dev.tsv)")
    parser.add_argument("-t", "--test_file", type=str, default='data/test.tsv',
                        help="The test set to read in (default data/test.tsv)")
    args = parser.parse_args()
    return args


def read_corpus(corpus_file):
    """Read in data set and returns docs and labels."""
    documents, labels = [], []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('\t')
            documents.append(tokens[0])
            labels.append(tokens[1])

    return documents, labels


def lr_schedule(epoch, initial_lr=5e-5, drop=0.5, epochs_drop=5):
    """Define a learning rate schedule."""
    return initial_lr * (drop ** (epoch // epochs_drop))


def experiment_with_model(tokens_train, tokens_dev, tokens_test, model_name, batch_size, epochs, Y_train_bin, Y_dev_bin,
                          Y_test_bin, encoder):
    """Load, train and evaluate a model."""
    # Load the model and tokenizer
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

    # Compile the model
    loss_function = CategoricalCrossentropy(from_logits=True)
    optim = Adam(learning_rate=5e-5)
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])

    # Learning rate scheduler (drops learning rate after a given number of epochs)
    lr_scheduler = LearningRateScheduler(lambda epoch: lr_schedule(epoch))

    # Train the model
    model.fit(tokens_train, Y_train_bin, validation_data=(tokens_dev, Y_dev_bin),
              epochs=epochs, batch_size=batch_size, callbacks=[EarlyStopping(patience=5), lr_scheduler])

    # Evaluate the model on the dev set
    Y_pred = model.predict(tokens_dev)["logits"]
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_dev_true = np.argmax(Y_dev_bin, axis=1)
    evaluate_model(Y_dev_true, Y_pred, encoder)

    # Evaluate the model on the test set
    Y_pred_test = model.predict(tokens_test)["logits"]
    Y_pred_test = np.argmax(Y_pred_test, axis=1)
    Y_test_true = np.argmax(Y_test_bin, axis=1)
    evaluate_model(Y_test_true, Y_pred_test, encoder)


def evaluate_model(Y_true, Y_pred, encoder):
    """Print a classification report and confusion matrix for generated labels."""
    # Classification report
    print('\n*** CLASSIFICATION REPORT ***')
    print(classification_report(Y_true, Y_pred))

    # Calculates and print the confusion matrix
    print('\n*** CONFUSION MATRIX ***')
    labels = encoder.classes_   # Use the original string labels (not one-hot encoded)
    cm = confusion_matrix(Y_true, Y_pred, labels=np.arange(len(labels)))
    print(' '.join(labels))
    print(cm)

    # This will plot the confusion matrix
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    #disp.plot(cmap=plt.cm.Blues)
    #plt.show()


def encode_data(Y_train, Y_dev, Y_test):
    """Transform string labels to one-hot encodings"""
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.transform(Y_dev)  # Note: use transform not fit_transform to avoid altering the encoder
    Y_test_bin = encoder.transform(Y_test)

    return Y_train_bin, Y_dev_bin, Y_test_bin, encoder


def tokenize_data(tokenizer, data_file, max_length):
    """Tokenize a data file with the specified tokenizer"""
    return tokenizer(data_file, padding=True, max_length=max_length, truncation=True, return_tensors="np").data


def main():
    args = create_arg_parser()

    # Retrieve commandline model parameters
    max_length = args.max_length
    batch_size = args.batch_size
    epochs = args.epochs

    # Read the data
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    X_test, Y_test = read_corpus(args.test_file)

    Y_train_bin, Y_dev_bin, Y_test_bin, encoder = encode_data(Y_train, Y_dev, Y_test)

    model_names = ["bert-base-uncased"]

    for model_name in model_names:
        # Tokenize the data
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens_train = tokenize_data(tokenizer, X_train, max_length)
        tokens_dev = tokenize_data(tokenizer, X_dev, max_length)
        tokens_test = tokenize_data(tokenizer, X_test, max_length)

        print(f"Experimenting with model: {model_name}")
        experiment_with_model(tokens_train, tokens_dev, tokens_test, model_name, batch_size, epochs, Y_train_bin,
                              Y_dev_bin, Y_test_bin, encoder)


if __name__ == "__main__":
    main()
