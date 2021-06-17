import os
import re
from tqdm import tqdm


def text_to_vw_fmt(text_sections_to_convert, data, vw_str):
    for pd_series in text_sections_to_convert:
        vw_str += f" |{pd_series} {data[pd_series]}"
    return vw_str


# extract this format: [word1, vector1[]], [word2, vector2[]], ...] from pandas df
def extract_from_df_row(list_of_words, list_of_vectors):
    section_word_vectors = []
    for word, vector in zip(list_of_words, list_of_vectors):
        word_vector = [f"{word}"]
        for element in vector:
            word_vector.append(f"{element}")    # should get a list with 300 elements
        section_word_vectors.append(word_vector)
        # should get a list with n number of elements, where n is number of words in that section
        # each element is a list with 300 elements
    return section_word_vectors


# incoming format: [word1_text, vector1[]], [word2_text, vector2[]], ... ]
# output format: ['|subj_d1', 'subj_word_1:d1', 'subj_word_n:d1' , '|subj_d2', 'subj_word_1_d2', ...,
#                 '|text_d1', 'text_word_1_d1', ...]
def reorder_word_vectors(section_word_vectors, section):
    section_vw = []
    vector_index = 1
    while vector_index != 301: # indexing starts at 1 because word_vect[0] == word_text
        section_vw.append(f"|{section}_d{vector_index}")
        for word_vect in section_word_vectors:
            section_vw.append(f"{word_vect[0]}:{word_vect[vector_index]}")
        vector_index += 1
    return section_vw


# goal: 1 |subj_d1 subj_word1:d1 subj_word2:d1 ... |subj_d2 subj_word1:d2 ... |text_d1 text_word1:d1 text_word2:d1 ...
def word_vectors_to_vw_fmt(word_vector_sections_to_convert, vw_str, data):
    for pd_series_text, pd_series_vectors in word_vector_sections_to_convert:
        word_rgx = re.compile(r'\w+')
        list_of_words = word_rgx.findall(data[pd_series_text])
        section = f"{pd_series_text}"

        # extract this format: [word1, vector1[]], [word2, vector2[]], ...] from pandas df
        section_word_vectors = extract_from_df_row(list_of_words, data[pd_series_vectors])

        # output format: ['|subj_d1', 'subj_word_1:d1', 'subj_word_n:d1' , '|subj_d2', 'subj_word_1_d2', ...,
        #                 '|text_d1', 'text_word_1_d1', ...]
        section_vw = reorder_word_vectors(section_word_vectors, section)

        # convert to str
        vw_seperator = " "
        section_vw_str = vw_seperator.join(section_vw)
        vw_str += f" {section_vw_str}"

    return vw_str


# incoming format: [[token_1_id, vector1[]], [token_2_id, vector2[]], ...]
# output format: ['|subj_d0', 'subj_word_1:d0', 'subj_word_n:d0' , '|subj_d1', 'subj_word_1:d1', ...,
#                 '|text_d1', 'text_word_1_d1', ...]
def reorder_subword_vectors(section_subword_vectors, section):
    section_vw = []
    vector_index = 0
    while vector_index != 300:
        section_vw.append(f"|{section}_d{vector_index}")
        for subword_vect in section_subword_vectors:
            section_vw.append(f"id{subword_vect[0]}:{subword_vect[1][vector_index]}")
        vector_index += 1

    return section_vw


# goal: 1 |subj_d1 subj_word1:d1 subj_word2:d1 ... |subj_d2 subj_word1:d2 ... |text_d1 text_word1:d1 text_word2:d1 ...
def subword_vectors_to_vw_fmt(subword_vector_sections_to_convert, vw_str, data):
    for pd_series in subword_vector_sections_to_convert:
        # input format: [[token_1_id, vector1[]], [token_2_id, vector2[]], ...] from pandas df

        # output format: ['|subj_d0', 'subj_word_1:d0', 'subj_word_n:d0' , '|subj_d1', 'subj_word_1:d1', ...,
        #                 '|text_d1', 'text_word_1_d1', ...]
        section_vw = reorder_subword_vectors(data[pd_series], pd_series)

        # convert to str
        vw_seperator = " "
        section_vw_str = vw_seperator.join(section_vw)
        vw_str += f" {section_vw_str}"

    return vw_str


# goal: 1 |subj_d1 d1, |subj_d2 d2, ... |text_d1 d1, |text_d2 d2, ...
def doc_vectors_to_vw_fmt(doc_vector_sections_to_convert, vw_str, data):
    for pd_series in doc_vector_sections_to_convert:
        vector_index = 1
        vw_str += f" |{pd_series}"
        for dim in data[pd_series]:
            vw_str += f" d{vector_index}:{dim}"
            vector_index += 1

    return vw_str


def store_vw_fmt(vw_str, filepath, filename):
    # write combined str to .txt file
    with open(os.path.join(filepath, filename), 'a') as storage_f:
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
# word_vector_sections_to_convert expects a list of tuples
# doc_vector_sections_to_convert expect a list
# subword_vector_sections_to_convert expects a list
def pd_to_vw_fmt(pd_df,
                 text_sections_to_convert, word_vector_sections_to_convert,
                 doc_vector_sections_to_convert, subword_vector_sections_to_convert,
                 filepath, filename, train):
    # convert each row in pandas df into vw format
    print('Çonverting to vw format...')
    if not train:
        test_label_filename = f"test_labels.txt"
        store_test_labels(filepath, test_label_filename, pd_df)
    # remove pre-existing file
    if os.path.isfile(os.path.join(filepath, filename)):
        os.remove(os.path.join(filepath, filename))
    for index, data in tqdm(pd_df.iterrows()):
        if train:
            vw_str = f"{data['target']}"
        else:
            vw_str = f""
        # convert only text data
        vw_str = text_to_vw_fmt(text_sections_to_convert, data, vw_str)
        # convert text with word embeddings
        vw_str = word_vectors_to_vw_fmt(word_vector_sections_to_convert, vw_str, data)
        # convert text with doc mean vector
        vw_str = doc_vectors_to_vw_fmt(doc_vector_sections_to_convert, vw_str, data)
        # convert text with subword embeddings
        vw_str = subword_vectors_to_vw_fmt(subword_vector_sections_to_convert, vw_str, data)
        # store as .txt file
        store_vw_fmt(vw_str, filepath, filename)
    print('Converted to vw format!')

    return 0

