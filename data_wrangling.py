# general python modules
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict


def load_from_pickle(filepath, filename):
    pickle_filepath = os.path.join(filepath, filename)
    if os.path.isfile(pickle_filepath):
        pd_df = pd.read_pickle(pickle_filepath)
        return pd_df

    else:
        print('No pre-existing file found')
        return -1


def remove_dup_words(spacy_nlp, pd_df, list_pd_series, filepath, filename, load_file=False):
    if not load_file:
        # remove pre-existing file
        if os.path.isfile(os.path.join(filepath, filename)):
            os.remove(os.path.join(filepath, filename))
        with spacy_nlp.select_pipes(enable='tokenizer'):
            for pd_series in list_pd_series:
                print(f"Removing dup words in {pd_series}...")
                unique_str_list = []
                for doc in tqdm(pd_df[pd_series]):
                    all_tokens = []
                    doc_unique_str = f""
                    for token in spacy_nlp(doc):
                        all_tokens.append(token.text)
                        unique_tokens = OrderedDict.fromkeys(all_tokens)

                    for key in unique_tokens:
                        doc_unique_str += f"{key} "

                    if len(doc_unique_str) == 0:
                        doc_unique_str = f" "
                    unique_str_list.append(doc_unique_str)
                pd_df.insert(loc=len(pd_df.columns), column=pd_series + '_unq', value=unique_str_list)

        # store unique words for later retrieval
        storage_f = os.path.join(filepath, filename)
        pd_df.to_pickle(path=storage_f)
        print('Unique words stored!')
        print('Unique words stored!')

    # load dataframe from memory
    if load_file:
        print('Loading unique words...')
        pd_df = load_from_pickle(filepath, filename)
        print('Unique words loaded!')
    return pd_df


def text_lemmatization(spacy_nlp, pd_df, list_pd_series, filepath, filename, load_file=False):
    if not load_file:
        # remove pre-existing file
        if os.path.isfile(os.path.join(filepath, filename)):
            os.remove(os.path.join(filepath, filename))
        with spacy_nlp.select_pipes(enable=['lemmatizer', 'tokenizer', 'tagger', 'attribute_ruler']):
            for pd_series in list_pd_series:
                print(f"Extracting {pd_series} lemmas...")
                compiled_lemmas = []
                for doc in tqdm(pd_df[pd_series]):
                    doc_lemmas = [token.lemma_ for token in spacy_nlp(doc)]
                    # converting from list to string
                    separator = ' '
                    doc_lemmas_string = separator.join(doc_lemmas)
                    # replace text in pandas df
                    compiled_lemmas.append(doc_lemmas_string)
                pd_df.insert(loc=len(pd_df.columns), column=pd_series+'_lem', value=compiled_lemmas)

        # store lemmatized text into a separate file for later retrieval
        storage_f = os.path.join(filepath, filename)
        pd_df.to_pickle(path=storage_f)
        print('Lemmatized text stored!')
        print('Lemmatized text loaded!')

    if load_file:
        print('Loading lemmatized text...')
        pd_df = load_from_pickle(filepath, filename)
        print('Lemmatized text loaded!')

    return pd_df


# create a null array, for handling of whitespace elements
def null_array(n):
    null_arr = np.zeros(n)
    return null_arr


def subword_embedding(bpemb_en, pd_df, list_pd_series, filepath, filename, load_file=False):
    if not load_file:
        # remove pre-existing file
        if os.path.isfile(os.path.join(filepath, filename)):
            os.remove(os.path.join(filepath, filename))
        for pd_series in list_pd_series:
            print(f"Extracting {pd_series} subword vectors...")
            compiled_vectors = []
            for doc in tqdm(pd_df[pd_series]):
                # finding word vectors
                token_ids = bpemb_en.encode_ids(doc)
                doc_vectors = bpemb_en.emb.vectors[token_ids]
                norm_vectors = []
                for vector, token_id in zip(doc_vectors, token_ids):
                    # normalize vector to [0, 1]
                    token_vector = ((vector / np.linalg.norm(vector)) + 1) / 2
                    # output format: [[token_1_id, vector1[]], [token_2_id, vector2[]], ...]
                    norm_vectors.append([token_id, token_vector])
                compiled_vectors.append(norm_vectors)
            # insert doc vectors in pandas df
            pd_df.insert(loc=len(pd_df.columns), column=pd_series+'_sw_vect', value=compiled_vectors)
        # store subword_vectors into a separate file for later retrieval
        storage_f = os.path.join(filepath, filename)
        pd_df.to_pickle(path=storage_f)
        print('\nSubword vectors loaded!')

    if load_file:
        print('Loading subword vectors...')
        pd_df = load_from_pickle(filepath, filename)
        print('Subword vectors loaded!')

    return pd_df


def doc_mean_vectors(bpemb_en, pd_df, list_pd_series, filepath, filename, load_file=False):
    if not load_file:
        # remove pre-existing file
        if os.path.isfile(os.path.join(filepath, filename)):
            os.remove(os.path.join(filepath, filename))
        for pd_series in tqdm(list_pd_series):
            print(f"Extracting {pd_series} mean vector...")
            compiled_mean_vectors = []
            for doc in pd_df[pd_series]:
                doc_vectors = bpemb_en.embed(doc)
                try:
                    mean_vectors = sum(doc_vectors) / len(doc_vectors)
                except ZeroDivisionError:
                    mean_vectors = null_array(300)
                compiled_mean_vectors.append(mean_vectors)
            # insert compiled_mean_vectors in pandas df
            pd_df.insert(loc=len(pd_df.columns), column=pd_series+'_m_vect', value=compiled_mean_vectors)
            print(f"{pd_series} mean vectors loaded!")
        # store doc_mean_vectors into a separate file for later retrieval
        storage_f = os.path.join(filepath, filename)
        pd_df.to_pickle(path=storage_f)
    if load_file:
        pd_df = load_from_pickle(filepath, filename)
        print('Mean vectors loaded!')

    return pd_df
