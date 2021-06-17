# general python modules
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
# project modules


# used for baseline bag_of_words model
def bag_of_words(document, label=None):
    return (
        str(label or "")
        + " |text "
        + " ".join(re.findall("\w{3,}", document.lower()))
        + "\n"
    )


def load_from_csv(filepath, filename):
    if os.path.isfile(os.path.join(filepath, filename)):
        temp_storage_list = []
        with open(os.path.join(filepath, filename), 'r') as storage_f:
            # loading in by chunks because of C error: out of memory due to large file size
            for chunk in pd.read_csv(storage_f, sep=',', chunksize=20000):
                temp_storage_list.append(chunk)
            pd_df = pd.concat(temp_storage_list, axis=0)
            del temp_storage_list
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
                print(f"Removing dup words in {pd_series}")
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
                pd_df.insert(loc=len(pd_df.columns), column=pd_series + '_unique', value=unique_str_list)
        with open(os.path.join(filepath, filename), 'w') as storage_f:
            pd_df.to_csv(path_or_buf=storage_f, index=False)
            print('Unique words stored!')
            print('Unique words stored!')

    # load dataframe from memory
    if load_file:
        print('Loading unique words...')
        pd_df = load_from_csv(filepath, filename)
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
                for doc in tqdm(pd_df[pd_series]):
                    doc_lemmas = [token.lemma_ for token in spacy_nlp(doc)]
                    # converting from list to string
                    separator = ' '
                    doc_lemmas_string = separator.join(doc_lemmas)
                    # replace text in pandas df
                    pd_df.loc[:, pd_series].replace(to_replace=doc, value=doc_lemmas_string, inplace=True)
                    # store lemmatized text into a separate file for later retrieval
        with open(os.path.join(filepath, filename), 'w') as storage_f:
            pd_df.to_csv(path_or_buf=storage_f, index=False)
            print('Lemmatized text stored!')
            print('Lemmatized text loaded!')

    # load dataframe from memory
    if load_file:
        print('Loading lemmatized text...')
        pd_df = load_from_csv(filepath, filename)
        print('Lemmatized text loaded!')
    return pd_df


# create a null array, for handling of whitespace elements
def null_array(n):
    null_arr = np.zeros(n)
    return null_arr


def word_embedding(spacy_nlp, pd_df, list_pd_series, filepath, filename, load_file=False):
    if not load_file:
        # remove pre-existing file
        if os.path.isfile(os.path.join(filepath, filename)):
            os.remove(os.path.join(filepath, filename))
        with spacy_nlp.select_pipes(enable=['tokenizer']):
            for pd_series in list_pd_series:
                print(f"Extracting {pd_series} word vectors...")
                compiled_vectors = []
                for doc in tqdm(pd_df[pd_series]):
                    doc_vectors = []
                    # finding word vectors
                    for token in spacy_nlp(doc):
                        if token.is_space or token.is_oov:
                            # preventing divide by zero error
                            doc_vectors.append(null_array(300))
                        else:
                            # normalize vector to [0, 1]
                            word_vector = ((token.vector / token.vector_norm) + 1) / 2
                            doc_vectors.append(word_vector)
                    compiled_vectors.append(doc_vectors)
                # insert doc vectors in pandas df
                pd_df.insert(loc=len(pd_df.columns), column=pd_series+'_w_vect', value=compiled_vectors)
            print('\nWord vectors loaded!')

    if load_file:
        print('Loading word vectors...')
        pd_df = load_from_csv(filepath, filename)
        print('Word vectors loaded!')

    return pd_df


def doc_mean_vectors(spacy_nlp, pd_df, list_pd_series, filepath, filename, load_file=False):
    if not load_file:
        # remove pre-existing file
        if os.path.isfile(os.path.join(filepath, filename)):
            os.remove(os.path.join(filepath, filename))
        with spacy_nlp.select_pipes(enable=['tokenizer']):
            for pd_series in tqdm(list_pd_series):
                print(f"Extracting {pd_series} mean vector...")
                compiled_mean_vectors = []
                for text in pd_df[pd_series]:
                    doc = spacy_nlp(text)
                    compiled_mean_vectors.append(doc.vector)
                # insert compiled_mean_vectors in pandas df
                pd_df.insert(loc=len(pd_df.columns), column=pd_series+'_m_vect', value=compiled_mean_vectors)
            print(f"{pd_series} mean vectors loaded!")

    if load_file:
        pd_df = load_from_csv(filepath, filename)
        print('Mean vectors loaded!')

    return pd_df
