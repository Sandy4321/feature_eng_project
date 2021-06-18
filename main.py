# general python modules
import os
# NLP modules
import spacy
# ML modules
from vowpalwabbit import pyvw
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from bpemb import BPEmb
import seaborn as sns
import matplotlib.pyplot as plt
# project modules
from vw_config import (
    target_dict,
    raw_data_fpath,
    bpemb_model_fpath,
    processed_data_fpath,
    pre_processed_filename,
    dups_removed_filename,
    pure_text_lemmatized_filename,
    words_vectorized_filename,
    doc_vectorized_filename,
    train_filename,
    test_filename,
    pred_fpath,
    pred_filename
)
from data_wrangling import text_lemmatization, word_embedding, doc_mean_vectors, remove_dup_words, subword_embedding
from data_pre_processing import sk_to_pd_df, text_pre_processing
from convert_to_vw_format import pd_to_vw_fmt
from evaluation_options import vw_opts, combinations_to_test


# WIP to work with new data format, combine individual data in VW format from each .txt file
# note: " |text_data_lem" <- whitespace char present at front of this
#       "|Subject_unq_sw_vect_d0" <- whitespace char absent at front of this, need to rerun code to fix :/
def train_model(vw_model, filepath, features):
    filenames_list = []
    features_list = []
    for feature in features:
        filenames_list.append(f"{feature}_train.txt")
    with open(os.path.join(filepath, 'train_data_labels.txt'), 'r') as label_file:
        labels = label_file.readlines()
    for filename in filenames_list:
        with open(os.path.join(filepath, filename[0]), 'r') as feature_file:
            features_list.append(feature_file.readlines())
    data_index = 0
    for label in labels:
        vw_str = f"{label}"
        for features in features_list:
            vw_str += f"{features[data_index]}"
        data_index += 1
        vw_model.learn(vw_str)

    return 0


def eval_model(vw_model, processed_data_fpath, filename, labels, pred_fpath, pred_filename):
    predictions = []
    with open(os.path.join(processed_data_fpath, filename)) as test_f:
        for sample in test_f.readlines():
            predictions.append(vw_model.predict(sample))

    # write results out to file
    if os.path.isfile(os.path.join(pred_fpath, pred_filename)):
        os.remove(os.path.join(pred_fpath, pred_filename))
    with open(os.path.join(pred_fpath, pred_filename), "a") as pred_file:
        for pred in predictions:
            pred_file.write(str(pred) + "\n")
    print("Classification report:")
    print(classification_report(labels, predictions))
    print("Accuracy score:")
    print(accuracy_score(labels, predictions))

    plot_cm = False
    if plot_cm:
        cm = confusion_matrix(labels, predictions)
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(
            cm, linewidth=0.5, annot=True, fmt=".0f", cmap="Blues", ax=ax, square=True
        )


def main():
    #  loading in data
    newsgroups = load_files(raw_data_fpath, encoding='ANSI')
    df = sk_to_pd_df(newsgroups, target_dict)

    # data processing
    spacy_nlp = spacy.load('en_core_web_md')
    bpemb_en = BPEmb(model_file=f"{bpemb_model_fpath}\\en.wiki.bpe.vs200000.model",
                     emb_file=f"{bpemb_model_fpath}\\en.wiki.bpe.vs200000.d300.w2v.bin", dim=300)

    # base pd_df sections: target | Subject | pure_text | text_data
    df = text_pre_processing(spacy_nlp, df, processed_data_fpath, pre_processed_filename,
                             load_file=True)

    # for BOW features
    # current pd_df sections: target | Subject | pure_text | text_data
    #                                | Subject_lem | pure_text_lem | text_data_lem
    df = text_lemmatization(spacy_nlp, df, ['Subject', 'pure_text', 'text_data'],
                            processed_data_fpath, pure_text_lemmatized_filename,
                            load_file=True)

    # creates new pd_series with key f"{pd_series}_unq"
    # for subword_vectors
    # current pd_df sections: target | Subject | pure_text | text_data
    #                                | Subject_lem | pure_text_lem | text_data_lem
    #                                | Subject_unq | pure_text_unq | text_data_unq
    df = remove_dup_words(spacy_nlp, df, ['Subject', 'pure_text', 'text_data'],
                          processed_data_fpath, dups_removed_filename, load_file=True)

    # creates new pd_series with key f"{pd_series}_m_vect"
    # current pd_df sections: target | Subject | pure_text | text_data
    #                                | Subject_lem | pure_text_lem | text_data_lem
    #                                | Subject_unq | pure_text_unq | text_data_unq
    #                                | Subject_unq_m_vect
    # df = doc_mean_vectors(spacy_nlp, df, ['Subject_unq'], processed_data_fpath, doc_vectorized_filename,
    #                      load_file=False)

    # creates new pd_series with key f"{pd_series}_sw_vect"
    # current pd_df sections: target | Subject | pure_text | text_data
    #                                | Subject_lem | pure_text_lem | text_data_lem
    #                                | Subject_unq | pure_text_unq | text_data_unq
    #                                | Subject_unq_m_vect
    #                                | Subject_unq_sw_vect | pure_text_unq_sw_vect | text_data_unq_sw_vect
    df = subword_embedding(bpemb_en, df, ['Subject_unq', 'pure_text_unq', 'text_data_unq'],
                           processed_data_fpath, words_vectorized_filename, load_file=True)

    # store training data
    data_train, data_test = train_test_split(df, test_size=0.25, random_state=1)
    pd_to_vw_fmt(pd_df=data_train,
                 text_sections_to_convert=['Subject_lem',
                                           'pure_text_lem',
                                           'text_data_lem'],
                 doc_vector_sections_to_convert=[],
                 subword_vector_sections_to_convert=['Subject_unq_sw_vect',
                                                     'pure_text_unq_sw_vect',
                                                     'text_data_unq_sw_vect'],
                 filepath=processed_data_fpath, train=True)
    # store testing data
    pd_to_vw_fmt(pd_df=data_test,
                 text_sections_to_convert=['Subject_lem',
                                           'pure_text_lem',
                                           'text_data_lem'],
                 doc_vector_sections_to_convert=[],
                 subword_vector_sections_to_convert=['Subject_unq_sw_vect',
                                                     'pure_text_unq_sw_vect',
                                                     'text_data_unq_sw_vect'],
                 filepath=processed_data_fpath, train=False)

    test_labels = data_test['target']  # for model evaluation

    # generate vw_model with corresponding params
    vw_model = pyvw.vw(**vw_opts)

    # train vw_model
    train_model(vw_model, processed_data_fpath, filename=train_filename)

    # evaluate vw_model against test set
    eval_model(
        vw_model,
        processed_data_fpath=processed_data_fpath,
        filename=test_filename,
        labels=test_labels,
        pred_fpath=pred_fpath,
        pred_filename=pred_filename,
    )

    return 0


if __name__ == "__main__":
    main()
