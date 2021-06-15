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
        # remove pre-existing file
        if os.path.isfile(os.path.join(filepath, filename)):
            os.remove(os.path.join(filepath, filename))
        with spacy_nlp.select_pipes(enable=['lemmatizer', 'tokenizer', 'tagger', 'attribute_ruler']):
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
        with spacy_nlp.select_pipes(enable=['tokenizer', 'tok2vec']):
            print('Extracting word vectors...')
            for pd_series in tqdm(list_pd_series):
                compiled_vectors = []
                for doc in pd_df[pd_series]:
                    doc_vectors = []
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
                pd_df.insert(loc=len(pd_df.columns), column=pd_series+'_vectors', value=compiled_vectors)
            print('Word vectors loaded!')

    if load_file:
        pd_df = load_from_csv(filepath, filename)
        print('Word vectors loaded!')

    return pd_df


def sentence_vectors(spacy_nlp, pd_df, list_pd_series):
    with spacy_nlp.select_pipes(enable=['tok2vec', 'parser']):
        print('Extracting sentence vectors...')
        for pd_series in tqdm(list_pd_series):
            compiled_sent_vectors = []
            for text in pd_df[pd_series]:
                doc = spacy_nlp(text)
                sentences = list(doc.sents)
                sentences_vectors = []
                for sentence in sentences:
                    if sentence.vector_norm == 0:
                        # preventing divide by zero error
                        sentences_vectors.append(null_array(300))
                    else:
                        # Vector normalisation into the range [0,1]
                        sent_vect = ((sentence.vector / sentence.vector_norm) + 1) / 2
                        sentences_vectors.append(sent_vect.astype("float64"))
                compiled_sent_vectors.append(sentences_vectors)
            # insert doc vectors in pandas df
            pd_df.insert(loc=len(pd_df.columns), column=pd_series+'_s_vect', value=compiled_sent_vectors)
        print('Sentence vectors loaded!')
    return pd_df


def doc_mean_vectors(spacy_nlp, pd_df, list_pd_series, filepath, filename, load_file=False):
    if not load_file:
        # remove pre-existing file
        if os.path.isfile(os.path.join(filepath, filename)):
            os.remove(os.path.join(filepath, filename))
        with spacy_nlp.select_pipes(enable=['tokenizer', 'tok2vec']):
            for pd_series in tqdm(list_pd_series):
                print(f"Extracting {pd_series} mean vector...")
                compiled_mean_vectors = []
                for text in pd_df[pd_series]:
                    doc = spacy_nlp(text)
                    if np.sum(doc.vector) == 0:
                        # preventing divide by zero error
                        doc_vector = null_array(300)
                    else:
                        # normalize vector to [0, 1]
                        doc_vector = ((doc.vector / doc.vector_norm) + 1) / 2
                    compiled_mean_vectors.append(doc_vector)
                # insert compiled_mean_vectors in pandas df
                pd_df.insert(loc=len(pd_df.columns), column=pd_series+'_m_vect', value=compiled_mean_vectors)
            print(f"{pd_series} mean vectors loaded!")

    if load_file:
        pd_df = load_from_csv(filepath, filename)
        print('Mean vectors loaded!')

    return pd_df
