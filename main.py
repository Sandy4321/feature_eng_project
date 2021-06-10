# general python modules
import os
# NLP modules
import spacy
# ML modules
from vowpalwabbit import pyvw
from sklearn.datasets import load_files
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# project modules
from vw_config import (
    vw_opts,
    raw_data_fpath,
    processed_data_fpath,
    pred_fpath,
    tokenized_filename,
    train_filename,
    test_filename,
    test_pred_filename,
)
from data_wrangling import bag_of_words, word_embeddings


def load_data(filepath):
    # load newsgroup data in
    all_train_newsgroups = load_files(filepath, encoding="ANSI")
    all_train_data = all_train_newsgroups["data"]
    topic_encoder = LabelEncoder()
    all_targets = topic_encoder.fit_transform(all_train_newsgroups["target"]) + 1
    # split data into train and test set
    train_docs, test_docs, train_labels, test_labels = train_test_split(
        all_train_data, all_targets, test_size=0.25, random_state=1
    )
    return train_docs, test_docs, train_labels, test_labels


def store_train_data_vw_format(filepath, filename, docs, labels):
    with open(os.path.join(filepath, filename), "w") as vw_data:
        for text, target in zip(docs, labels):
            vw_data.write(bag_of_words(text, target))
    return 0


def store_eval_data_vw_format(filepath, filename, docs):
    with open(os.path.join(filepath, filename), "w") as vw_data:
        for text in docs:
            vw_data.write(bag_of_words(text))
    return 0


def train_model(vw_model, filepath, filename):
    with open(os.path.join(filepath, filename), "r") as train_f:
        train_examples = train_f.readlines()
        for example in train_examples:
            vw_model.learn(example)
    return 0


def eval_model(vw_model, processed_data_fpath, filename, labels, pred_fpath, pred_filename):
    predictions = []
    with open(os.path.join(processed_data_fpath, filename)) as f:
        for sample in f.readlines():
            predictions.append(vw_model.predict(sample))

    # write results out to file
    if os.path.isfile(os.path.join(pred_fpath, pred_filename)):
        os.remove(os.path.join(pred_fpath, pred_filename))
    with open(os.path.join(pred_fpath, pred_filename), "a") as pred_file:
        for pred in predictions:
            pred_file.write(str(pred) + "\n")
    print("Classification report:")
    print(classification_report(labels, predictions))

    plot_cm = False
    if plot_cm:
        cm = confusion_matrix(labels, predictions)
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(
            cm, linewidth=0.5, annot=True, fmt=".0f", cmap="Blues", ax=ax, square=True
        )


def main():
    #  loading in data
    train_docs, test_docs, train_labels, test_labels = load_data(raw_data_fpath)

    # data processing
    spacy_nlp = spacy.load('en_core_web_md')
    word_embeddings(spacy_nlp, docs=train_docs, filepath=processed_data_fpath, filename=tokenized_filename)
    """
    # store training data
    store_train_data_vw_format(
        processed_data_fpath, filename=train_filename, docs=train_docs, labels=train_labels
    )
    # store testing data
    store_eval_data_vw_format(processed_data_fpath, filename=test_filename, docs=test_docs)
    
    # generate vw_model with corresponding params
    vw_model = pyvw.vw(**vw_opts)

    # train vw_model
    train_model(vw_model, processed_data_fpath, filename=train_filename)

    # evaluate vw_model against test set
    eval_model(
        vw_model,
        processed_data_fpath = processed_data_fpath,
        filename=test_filename,
        labels=test_labels,
        pred_fpath = pred_fpath,
        pred_filename=test_pred_filename,
    )
    """
if __name__ == "__main__":
    main()