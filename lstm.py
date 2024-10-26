
#!/usr/bin/env python

"""
LfD Final Assignment - LSTM models

This script allows the user to run an LSTM classifier on a binary classification task.

Available command-line options:
-i  Input file to learn from (default data/train.tsv)
-d  Separate dev set to read in (default data/dev.tsv)
-t  If added, use trained model to predict on test set
-e  Embedding file we are using (default data/cc.en.300.vec)
-m  Use more data (the MHS dataset)

"""

import random as python_random
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.layers import TextVectorization
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
import string
import datasets
import re
from keras.initializers import Constant
from collections import Counter
# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='data/train.tsv', type=str,
                        help="Input file to learn from (default data/train.tsv)")
    parser.add_argument("-d", "--dev_file", type=str, default='data/dev.tsv',
                        help="Separate dev set to read in (default data/dev.tsv)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    parser.add_argument("-e", "--embeddings", default='data/cc.en.300.vec', type=str,
                        help="Embedding file we are using (default data/cc.en.300.vec)")
    parser.add_argument("-m", "--mhs", action="store_true",
                        help="Use more OFF training data, MHS")
    parser.add_argument("--hsuse", action="store_true",
                        help="Use more OFF training data, HSUSE")
    args = parser.parse_args()
    return args


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
    """Read in data set and returns docs and labels"""
    documents, labels = [], []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('\t')
            documents.append(tokens[0].translate(str.maketrans('', '', string.punctuation)))
            labels.append(tokens[1])

    return documents, labels


def read_embeddings(embeddings_file):
    """Read in word embeddings from file and save as numpy array"""
    with open(embeddings_file, 'r') as f:
        embeddings = f.readlines()
    return {line.split()[0]: np.array(line.split()[1:]) for line in embeddings}


def get_emb_matrix(voc, emb):
    """Get embedding matrix given vocab and the embeddings"""
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def create_model(Y_train, emb_matrix):
    """Create an LSTM model using Keras"""
    # Define settings
    learning_rate = 0.001
    loss_function = 'binary_crossentropy'
    optim = RMSprop(learning_rate=learning_rate)
    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])

    num_tokens = len(emb_matrix)
    num_labels = len(set(Y_train))

    # Now build the model
    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=False))

    # LSTM layers
    model.add(Bidirectional(LSTM(units=num_labels, dropout=0.45, activation="elu")))

    # Compile model using our settings, check for accuracy
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev, encoder, args):
    """Train the model here. Note the different settings you can experiment with!"""
    # Hyperparams
    verbose = 1
    batch_size = 32
    epochs = 50
    # Early stopping: stop training when there are three consecutive epochs without improving
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # Finally fit the model to our data
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[callback], batch_size=batch_size, validation_data=(X_dev, Y_dev))
    # Print final accuracy for the model
    test_set_predict(model, X_dev, Y_dev, "dev", encoder, args)
    return model


def test_set_predict(model, X_test, Y_test, ident, encoder, args):
    """Do predictions and measure accuracy on our own test set (that we split off train)"""
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)

    boundary = 0.65

    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = (Y_pred > boundary).astype("int32")

    print('Accuracy on own {1} set: {0}'.format(round(accuracy_score(Y_test, Y_pred), 3), ident))

    # Print detailed results
    print_evaluation(Y_test, Y_pred, encoder)


def print_evaluation(Y_test, Y_pred, encoder):
    """Takes true labels and predicted labels and
    prints evaluation measures (a classification report
    and confusion matrix)"""
    labels = encoder.classes_

    print('\n*** CLASSIFICATION REPORT ***')
    print(classification_report(Y_test, Y_pred, digits=3))

    print('\n*** CONFUSION MATRIX ***')
    cm = confusion_matrix(Y_test, Y_pred)
    print(' '.join(labels))
    print(cm)

    # Pretty confusion matrix
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    #disp.plot(cmap=plt.cm.Blues)
    #plt.show()

def main():
    """Main function to train and test neural network given cmd line arguments"""
    args = create_arg_parser()

    # Read in the data and embeddings
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    embeddings = read_embeddings(args.embeddings)

    # Create additional OFF data if argument is given
    if args.mhs:
        X_train_aug, Y_train_aug = create_new_data('mhs')
        X_train = X_train + X_train_aug
        Y_train = Y_train + Y_train_aug
    elif args.hsuse:
        X_train_aug, Y_train_aug = create_new_data('hsuse')
        X_train = X_train + X_train_aug
        Y_train = Y_train + Y_train_aug

    print('Train label distribution: ', Counter(Y_train))
    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)

    # Create model
    model = create_model(Y_train, emb_matrix)

    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    # Train the model
    model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin, encoder, args)

    # Do predictions on specified test set
    if args.test_file:
        # Read in test set and vectorize
        X_test, Y_test = read_corpus(args.test_file)
        Y_test_bin = encoder.fit_transform(Y_test)
        X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
        # Finally do the predictions
        test_set_predict(model, X_test_vect, Y_test_bin, "test", encoder, args)


if __name__ == '__main__':
    main()
