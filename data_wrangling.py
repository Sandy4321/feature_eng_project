# general python modules
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
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


def text_lemmatization(spacy_nlp, pd_df, pd_series, filepath, filename, load_file=False):
    if not load_file:
        with spacy_nlp.select_pipes(enable=['lemmatizer', 'tokenizer', 'tagger', 'attribute_ruler']):
            for doc in tqdm(pd_df[pd_series]):
                doc_lemmas = [token.lemma_ for token in spacy_nlp(doc)]
                # converting from list to string
                separator = ' '
                doc_lemmas_string = separator.join(doc_lemmas)
                # replace text in pandas df
                pd_df[pd_series].replace(to_replace=doc, value=doc_lemmas_string, inplace=True)
                # store lemmatized text into a separate file for later retrieval
            with open(os.path.join(filepath, filename), 'w') as storage_f:
                pd_df.to_csv(path_or_buf=storage_f, index=False)

    # load dataframe from memory
    if load_file:
        pd_df = load_from_csv(filepath, filename)

    return pd_df


# create a null array, for handling of whitespace elements
def null_array(n):
    null_arr = np.zeros(n)
    return null_arr


def word_embedding(spacy_nlp, pd_df, list_pd_series, filepath, filename, load_file=False):
    if not load_file:
        with spacy_nlp.select_pipes(enable=['tokenizer', 'tok2vec']):
            for pd_series in tqdm(list_pd_series):
                compiled_vectors = []
                for doc in tqdm(pd_df[pd_series]):
                    doc_vectors = []
                    for token in spacy_nlp(doc):
                        if token.is_space:
                            doc_vectors.append(null_array(300))
                        else:
                            doc_vectors.append(token.vector)
                    compiled_vectors.append(doc_vectors)
                # insert doc vectors in pandas df
                pd_df.insert(loc=len(pd_df.columns), column=pd_series+'_vectors', value=compiled_vectors)
    if load_file:
        pd_df = load_from_csv(filepath, filename)

    return pd_df
