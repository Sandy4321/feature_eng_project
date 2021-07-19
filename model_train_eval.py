import gc
import os
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, f1_score
from vowpalwabbit import pyvw
from hyperopt import tpe, hp, fmin, Trials
from math import log
import pandas as pd


# input label format example: '5'
# input feature format example (vect): '|Subject_unq_sw_vect_d0 id19:0.563 ... |Subject_unq_sw_vect_d1 id32:0.521 ...'
# input feature format example (text): '|text_data_lem year biggest worst ...'
# notes about input feature format:
# 1) whitespace absent at front and end of string
# 2) features are already in vw format, only need to append to current vw line
# output to vw format: '[label] [feature_1] [feature_2] ...'
def to_vw(processed_data_fpath, features, train):
    strip_newline = re.compile(r'\n')
    if train:
        filenames_list = ['train_data_labels.txt']
        for feature in features:
            filenames_list.append(f"{feature}_train.txt")
    elif not train:
        filenames_list = []
        for feature in features:
            filenames_list.append(f"{feature}_test.txt")
    file_list = [open(os.path.join(processed_data_fpath, filename), 'r') for filename in filenames_list]
    vw_str_list = []
    for feature_tuple in tqdm(zip(*file_list)):
        vw_str = ''.join(feature_tuple)
        vw_str = re.sub(strip_newline, ' ', vw_str)
        vw_str_list.append(vw_str)

    return vw_str_list


# train vw_model
def train_model(vw_model, train_str_list):
    for train_vw_str in train_str_list:
        vw_model.learn(train_vw_str)

    return 0


# evaluate vw_model against test set
def eval_model(vw_model, test_str_list):
    predictions = []
    for sample in test_str_list:
        predictions.append(vw_model.predict(sample))
    return predictions


# evaluate vw_model against test set for hyperopt
def eval_model_hyperopt(vw_model, test_str_list, labels):
    predictions = []
    for sample in test_str_list:
        predictions.append(vw_model.predict(sample))
    return accuracy_score(labels, predictions), f1_score(labels, predictions, average='weighted')


# write results out to file
def store_results(features, pred_fpath, pred_filename, labels, predictions, ngrams, run):
    if os.path.isfile(os.path.join(pred_fpath, f"{pred_filename}_n{ngrams}_{run}.txt")):
        os.remove(os.path.join(pred_fpath, f"{pred_filename}_n{ngrams}_{run}.txt"))
    with open(os.path.join(pred_fpath, f"{pred_filename}_n{ngrams}_{run}.txt"), 'a') as pred_file:
        for pred in predictions:
            pred_file.write(str(pred) + '\n')

    # store classification report
    report = classification_report(labels, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    classification_report_fname = f"Classification_report_n{ngrams}_{run}"
    report_df.to_csv(os.path.join(pred_fpath, f"{classification_report_fname}.txt"))

    # store features and accuracy score
    with open(os.path.join(pred_fpath, f"{classification_report_fname}_info.txt"), 'a') as f:
        features_str = ', '.join(features)
        f.write(f"Features: {features_str}")
        f.write(f"\nAccuracy Score = ")
        f.write(str(accuracy_score(labels, predictions)))
        f.write(f"\nf1_score = ")
        f.write(str(f1_score(labels, predictions, average='weighted')))

    print('Classification report:')
    print(classification_report(labels, predictions))
    print('Accuracy score:')
    print(accuracy_score(labels, predictions))
    print('f1 score:')
    print(f1_score(labels, predictions, average='weighted'))


def store_results_hyperopt(features, acc_score, best, pred_fpath, run):
    try:
        ngrams = best['ngram']
    except KeyError:
        ngrams = 0

    classification_report_fname = f"Classification_report_n{ngrams}_{run}"
    # store features and accuracy score
    with open(os.path.join(pred_fpath, f"{classification_report_fname}_info.txt"), 'a') as f:
        features_str = ', '.join(features)
        f.write(f"Features: {features_str}\n")
        f.write(f"Accuracy Score = {acc_score}\n")
        f.write(f"VW_model_opts: {best}")


def model_train_test_hyperopt(train_str_list, test_str_list, processed_data_fpath,
                              features, pred_fpath, run=1):
    with open(os.path.join(processed_data_fpath, 'test_data_labels.txt')) as label_file:
        labels = label_file.readlines()
        labels = list(map(int, labels))

    # Initialize trials object
    trials = Trials()
    # only use ngrams when features don't include vectors
    use_space_1 = True
    for feature in features:
        if 'vect' in feature:
            use_space_1 = False
            break
    if use_space_1:
        space = {
            # General options
            'random_seed': 7,
            # Output options
            'progress': 1000,
            # Example Manipulation options
            'ngram': hp.choice('ngram', [1, 2, 3]),
            # Update rule options
            'l1': hp.loguniform('l1', log(1e-8), log(1e-1)),
            'l2': hp.loguniform('l2', log(1e-8), log(1e-1)),
            'loss_function': 'logistic',
            # Weight options
            'bit_precision': 28,
            # Multiclass options
            'oaa': 20
        }
    else:
        space = {
            # General options
            'random_seed': 7,
            # Output options
            'progress': 1000,
            # Update rule options
            'l1': hp.loguniform('l1', log(1e-8), log(1e-1)),
            'l2': hp.loguniform('l2', log(1e-8), log(1e-1)),
            'loss_function': 'logistic',
            # Weight options
            'bit_precision': 28,
            # Multiclass options
            'oaa': 20
        }

    # hyperopt to find highest accuracy score
    def objective(space):
        vw_model = pyvw.vw(**space)
        train_model(vw_model, train_str_list)
        acc_score, f1_score_value = eval_model_hyperopt(vw_model, test_str_list, labels)
        del vw_model
        gc.collect()
        return -acc_score

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=2,
        trials=trials
    )

    print("Best: {}".format(best))
    print(-min(trials.losses()))
    store_results_hyperopt(features, -min(trials.losses()), best, pred_fpath, run)

    return 0
