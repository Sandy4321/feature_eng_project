import os
from tqdm import tqdm
from project_constants import pd_series_first_char_dict


# output format: |pd_series word1 word2 ...
def text_to_vw_fmt(pd_series, data, namespace_1st_char):
    vw_str = f"|{namespace_1st_char}_{pd_series} {data[pd_series]}"
    return vw_str


# incoming format: [[token_1_id, vector1[]], [token_2_id, vector2[]], ...]
# output format: ['|subj_d0', 'subj_word_1:d0', 'subj_word_n:d0' , '|subj_d1', 'subj_word_1:d1', ...
def reorder_subword_vectors(subword_vect_list, section, namespace_1st_char):
    section_vw = []
    vector_index = 0
    while vector_index != 300:
        section_vw.append(f"|{namespace_1st_char}_{section}_d{vector_index}")
        # subword_vect = [token_1_id, vector1[]]
        for subword_vect in subword_vect_list:
            section_vw.append(f"id{subword_vect[0]}:{subword_vect[1][vector_index]}")
        vector_index += 1

    return section_vw


# output format: 1 |subj_d1 subj_word1:d1 subj_word2:d1 ... |subj_d2 subj_word1:d2 ...
def subword_vectors_to_vw_fmt(pd_series, data, namespace_1st_char):
    # input format: [[token_1_id, vector1[]], [token_2_id, vector2[]], ...] from pandas df

    # output format: ['|subj_d0', 'subj_word_1:d0', 'subj_word_n:d0' , '|subj_d1', 'subj_word_1:d1', ...,
    section_vw = reorder_subword_vectors(data[pd_series], pd_series, namespace_1st_char)

    # convert to str
    vw_seperator = ' '
    vw_str = vw_seperator.join(section_vw)

    return vw_str


# output format: 1 |subj_d1 d1, |subj_d2 d2, ...
def doc_vectors_to_vw_fmt(pd_series, data, namespace_1st_char):
    vector_index = 1
    vw_str = f"|{namespace_1st_char}_{pd_series}"
    for dim in data[pd_series]:
        vw_str += f" d{vector_index}:{dim}"
        vector_index += 1

    return vw_str


def store_vw_fmt(vw_str, filepath, filename, train):
    # write combined str to .txt file
    if train:
        storage_filename = f"{filename}_train.txt"
    else:
        storage_filename = f"{filename}_test.txt"

    with open(os.path.join(filepath, storage_filename), 'a') as storage_f:
        storage_f.write(f"{vw_str}\n")


def store_test_labels(filepath, test_label_filename, pd_df):
    # remove pre-existing file
    if os.path.isfile(os.path.join(filepath, test_label_filename)):
        os.remove(os.path.join(filepath, test_label_filename))
    test_labels = pd_df['target']
    with open(os.path.join(filepath, test_label_filename), 'a') as test_labels_f:
        for label in test_labels:
            test_labels_f.write(f"{label}\n")
    return 0


# note: for arguments,
# text_sections_to_convert expect a list
# doc_vector_sections_to_convert expect a list
# subword_vector_sections_to_convert expects a list
def pd_to_vw_fmt(pd_df,
                 text_sections_to_convert,
                 doc_vector_sections_to_convert, subword_vector_sections_to_convert,
                 filepath, train):

    print('Converting to vw format...')
    # storing labels, change to iterate with loop below instead
    if train:
        label_filename = 'train_data_labels.txt'
    else:
        label_filename = 'test_data_labels.txt'
    with open(os.path.join(filepath, label_filename), 'w') as label_file:
        pd_df['target'].to_csv(path_or_buf=label_file, index=False, header=False, line_terminator='\n')

    for index, data in tqdm(pd_df.iterrows()):
        # convert only text data
        for pd_series in text_sections_to_convert:
            try:  # make 1st char of each namespace unique for vw
                namespace_1st_char = pd_series_first_char_dict[pd_series]
            except KeyError:
                return f"{pd_series} does not exist"
            vw_str = text_to_vw_fmt(pd_series, data, namespace_1st_char)
            store_vw_fmt(vw_str, filepath, pd_series, train)

        # convert text with doc mean vector
        for pd_series in doc_vector_sections_to_convert:
            try:  # make 1st char of each namespace unique for vw
                namespace_1st_char = pd_series_first_char_dict[pd_series]
            except KeyError:
                return f"{pd_series} does not exist"
            vw_str = doc_vectors_to_vw_fmt(pd_series, data, namespace_1st_char)
            store_vw_fmt(vw_str, filepath, pd_series, train)

        # convert text with subword embeddings
        for pd_series in subword_vector_sections_to_convert:
            try:  # make 1st char of each namespace unique for vw
                namespace_1st_char = pd_series_first_char_dict[pd_series]
            except KeyError:
                return f"{pd_series} does not exist"
            vw_str = subword_vectors_to_vw_fmt(pd_series, data, namespace_1st_char)
            store_vw_fmt(vw_str, filepath, pd_series, train)

    print('Converted to vw format!')

    return 0
