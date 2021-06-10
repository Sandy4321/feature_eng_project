import os
import re
from tqdm import tqdm


def text_to_vw_fmt(text_sections_to_convert, train, data):
    if train:
        vw_str = f"{data['target']}"
    else:
        vw_str = f""
    for pd_series in text_sections_to_convert:
        vw_str = f"{vw_str} |{pd_series} {data[pd_series]}"
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
# output format: ['|subj, 'word1:d0', ...'word1:d299', ... ,'word2:d0', ..., '|text', 'word1:d0' ...]
def reorder_word_vectors(section_word_vectors, section):
    section_vw = []
    vector_index = 1
    section_vw.append(f"|{section}")
    while vector_index != 301:  # indexing starts at 1 because word_vect[0] == word_text
        for word_vect in section_word_vectors:
            section_vw.append(f"{word_vect[0]}{vector_index}:{word_vect[vector_index]}")
        vector_index += 1
    return section_vw


# goal: 1 |subj_d1 subj_word1:d1 subj_word2:d1 ... |subj_d2 subj_word1:d2 ... |text_d1 text_word1:d1 text_word2:d1 ...
def vectors_to_vw_fmt(vector_sections_to_convert, vw_str, data):
    for pd_series_text, pd_series_vectors in vector_sections_to_convert:
        word_rgx = re.compile(r'\w+')
        list_of_words = word_rgx.findall(data[pd_series_text])
        section = f"{pd_series_text}"

        # extract this format: [word1, vector1[]], [word2, vector2[]], ...] from pandas df
        section_word_vectors = extract_from_df_row(list_of_words, data[pd_series_vectors])

        # output format: ['|subj, 'word1:d0', ...'word1:d299', ... ,'word2:d0', ..., '|text', 'word1:d299', ...]
        section_vw = reorder_word_vectors(section_word_vectors, section)

        # convert to str
        vw_seperator = " "
        section_vw_str = vw_seperator.join(section_vw)
        vw_str = f"{vw_str} {section_vw_str}"

    return vw_str


def store_vw_fmt(vw_str, filepath, filename):
    # write combined str to .txt file
    with open(os.path.join(filepath, filename), 'a') as storage_f:
        storage_f.write(f"{vw_str}\n")


# note: text_sections_to_convert expect a list as argument, vector_sections_to_convert expects a tuple or list
def pd_to_vw_fmt(pd_df, text_sections_to_convert, vector_sections_to_convert, filepath, filename, train):
    # convert each row in pandas df into vw format
    for index, data in tqdm(pd_df.iterrows()):
        # convert only text data
        vw_str = text_to_vw_fmt(text_sections_to_convert, train, data)
        # convert text with word embeddings
        vw_str = vectors_to_vw_fmt(vector_sections_to_convert, vw_str, train, data)
        # store as .txt file
        store_vw_fmt(vw_str, filepath, filename)
