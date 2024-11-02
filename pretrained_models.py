#!/usr/bin/env python

"""
LfD Final Assignment - Pretrained Models

This script allows the user to run several Pretrained Models on a binary classification task.

-e Use more OFF training data, HSUSE
-m Use more OFF training data, MHS

Necessary versions:
!pip install tensorflow==2.15.0
!pip install tf-keras==2.15.0
!pip install transformers==4.37.0
"""

import random
import numpy as np
import argparse
import string
import re
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Make reproducible
np.random.seed(1234)
tf.random.set_seed(1234)
random.seed(1234)


def create_arg_parser():
    """Creates argumentparser and defines command-line options that can be called upon."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mhs", action="store_true",
                        help="Use more OFF training data, MHS")
    parser.add_argument("-e", "--hsuse", action="store_true",
                        help="Use more OFF training data, HSUSE")

    return parser.parse_args()


def create_new_data(type):
    """Create additional offensive data"""
    documents, labels = [], []

    if type == 'mhs':
        # Retrieve dataset
        dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')
        df = dataset['train'].to_pandas()

        # Retrieve two relevant columns
        augment_data = df[['hate_speech_score', 'text']]
        augment_data = augment_data.values.tolist()

        # Create 4000 lines of new data and save
        counter = 0

        for line in augment_data:
            if counter < 4000 and line[0] >= 0.5:
                counter += 1
                cleaned = re.sub('@\w+', '@USER', line[1])
                cleaned = re.sub('https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', 'URL', cleaned)
                documents.append(cleaned.translate(str.maketrans('', '', string.punctuation)))
                labels.append('OFF')

    elif type == 'hsuse':
        counter = 0
        with open('data/hsus_train.tsv', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split('\t')
                if tokens[4] == 'Hateful' and counter < 4000:
                    counter += 1
                    cleaned = re.sub('@\w+', '@USER', tokens[0])
                    cleaned = re.sub('https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', 'URL', cleaned)
                    documents.append(cleaned.translate(str.maketrans('', '', string.punctuation)))
                    labels.append('OFF')
    return documents, labels


def read_corpus(corpus_file):
    """Define a function to read the corpus and remove punctuation"""
    documents, labels = [], []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('\t')
            doc = tokens[0].translate(str.maketrans('', '', string.punctuation))
            documents.append(doc)
            labels.append(tokens[1])

    return documents, labels


def tokenize_data(tokenizer, data, max_length):
    """Function to tokenize data"""
    return tokenizer(data, padding=True, max_length=max_length, truncation=True, return_tensors="np").data



def lr_schedule(epoch, initial_lr=5e-5, drop=0.5, epochs_drop=5):
    """Define a learning rate schedule function"""
    return initial_lr * (drop ** (epoch // epochs_drop))


def experiment_with_model(args, model_name, max_length, batch_size, epochs):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Read the data
    X_train, Y_train = read_corpus('data/train.tsv')
    X_dev, Y_dev = read_corpus('data/dev.tsv')
    X_test, Y_test = read_corpus('data/test.tsv')

    # Create additional OFF data if argument is given
    if args.mhs:
        X_train_aug, Y_train_aug = create_new_data('mhs')
        X_train = X_train + X_train_aug
        Y_train = Y_train + Y_train_aug
    elif args.hsuse:
        X_train_aug, Y_train_aug = create_new_data('hsuse')
        X_train = X_train + X_train_aug
        Y_train = Y_train + Y_train_aug

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.fit_transform(Y_dev)
    Y_test_bin = encoder.fit_transform(Y_test)

    # Tokenize the data
    tokens_train = tokenize_data(tokenizer, X_train, max_length)
    tokens_dev = tokenize_data(tokenizer, X_dev, max_length)
    tokens_test = tokenize_data(tokenizer, X_test, max_length)

    # Compile the model
    optim = Adam(learning_rate=5e-5)
    model.compile(optimizer=optim, metrics=['accuracy'])

    # Learning rate scheduler
    lr_scheduler = LearningRateScheduler(lambda epoch: lr_schedule(epoch))

    # Train the model
    history = model.fit(tokens_train, Y_train_bin, validation_data=(tokens_dev, Y_dev_bin),
                        epochs=epochs, batch_size=batch_size, callbacks=[EarlyStopping(patience=3), lr_scheduler])

    # Evaluate the model on the test set
    Y_pred = model.predict(tokens_test)["logits"]
    Y_pred = np.argmax(Y_pred, axis=1)

    Y_test_true = Y_test_bin
    accuracy = accuracy_score(Y_test_true, Y_pred)
    f1 = f1_score(Y_test_true, Y_pred, average='macro')

    cm = confusion_matrix(Y_test_true, Y_pred)
    print(cm)

    print(classification_report(Y_test_true, Y_pred, digits=3))

    print(f'Accuracy on validation set with model {model_name}: {accuracy:.3f}')
    print('Macro f1 on validation set with model:  ', f1)
    return history


def main():
    # Create argument parser
    args = create_arg_parser()

    # Define model names
    model_names = ["google-bert/bert-base-uncased"]
    results = {}

    for model_name in model_names:
        print(f"Experimenting with model: {model_name}")
        history = experiment_with_model(args, model_name=model_name, max_length=128, batch_size=32, epochs=5)
        results[model_name] = history


if __name__ == '__main__':
    main()