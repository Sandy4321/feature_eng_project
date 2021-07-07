import os
import re
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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


def train_model(vw_model, train_str_list):
    for train_vw_str in train_str_list:
        vw_model.learn(train_vw_str)

    return 0


def eval_model(vw_model, test_str_list, processed_data_fpath, features, pred_fpath, pred_filename, random_state, run=1):
    predictions = []
    for sample in test_str_list:
        predictions.append(vw_model.predict(sample))
    with open(os.path.join(processed_data_fpath, 'test_data_labels.txt')) as label_file:
        labels = label_file.readlines()
        labels = list(map(int, labels))

    # write results out to file
    if os.path.isfile(os.path.join(pred_fpath, f"{pred_filename}_r{random_state}_{run}.txt")):
        os.remove(os.path.join(pred_fpath, f"{pred_filename}_r{random_state}_{run}.txt"))
    with open(os.path.join(pred_fpath, f"{pred_filename}_r{random_state}_{run}.txt"), 'a') as pred_file:
        for pred in predictions:
            pred_file.write(str(pred) + '\n')

    # store classification report
    report = classification_report(labels, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    classification_report_fname = f"Classification_report_r{random_state}_{run}"
    report_df.to_csv(os.path.join(pred_fpath, f"{classification_report_fname}.txt"))

    # store features and accuracy score
    with open(os.path.join(pred_fpath, f"{classification_report_fname}_info.txt"), 'a') as f:
        features_str = ', '.join(features)
        f.write(f"Features: {features_str}")
        f.write(f"\nAccuracy Score = ")
        f.write(str(accuracy_score(labels, predictions)))
    print('Classification report:')
    print(classification_report(labels, predictions))
    print('Accuracy score:')
    print(accuracy_score(labels, predictions))
