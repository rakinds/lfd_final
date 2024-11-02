# LfD Final Project - Offensive Language Classification

This repository contains the code for the Learning from Data final project. Three different types of models can be run on a binary offensive language classification task. 

## Datasets

The necessary datasets can be found below. These need to be downloaded, preprocessed and added to a `data` folder to be used.

| Dataset    | Link      | Comments |
| ------------- | ------------- | ------------- |
| Base | https://github.com/idontflow/OLID | Only the 'tweet' and 'subtask_a' columns are relevant |
| MHS | https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech | This dataset is loaded automatically in the code |
| HSUSE |  https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/stance-hof/ | Only the  'text' and 'HOF' columns are relevant |
| OL | https://github.com/AmitDasRup123/OffensiveLang | Only the 'Text' and 'Final Annotation' columns are relevant |

## Running the models

### Naive Bayes
Available command-line options:

* -t Define the train file to use. Uses data/train.tsv default.
* -d Define the dev / test file to use. Uses data/dev.tsv default
* -p Displays a plotted version of the confusion matrix for the report. May not always work
                             from the commandline if display packages are not installed.
* -a Augmented: run the NB model with augmented features
* -o Use more OffensiveLang training data
* -e Use more hate speech in US elections training data
* -m Use more MHS training data

Usage example:

`python3 classic_models.py -o -a`

### Bidirectional LSTM
Available command-line options:

* -i  Input file to learn from (default data/train.tsv)
* -d  Separate dev set to read in (default data/dev.tsv)
* -t  If added, use trained model to predict on test set
* -e  Embedding file we are using (default data/cc.en.300.vec)
* -o  Use more training data (the OffensiveLang dataset)
* --hsuse Use more training data (HSUSE)
* -m Use more MHS training data

Usage example:

`python3 lstm.py -m`

### Pretrained models
Available command-line options:

* -e Use more OFF training data, HSUSE (Hate Speech in US Elections)
* -o Use more OFF training data, OL (OffensiveLang)
* -m Use more OFF training data, MHS (Measuring Hate Speech)

Usage example:

`python3 pretrained_models.py -e`

