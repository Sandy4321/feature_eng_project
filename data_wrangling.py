import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def bag_of_words(document, label=None):
    return (
        str(label or "")
        + " |text "
        + " ".join(re.findall("\w{3,}", document.lower()))
        + "\n"
    )


def extract_metadata(document):
    # seperate ("From", "Subject") lines from text
    from_pattern = re.compile(r'(?<=From:).*\s?')
    subject_pattern = re.compile(r'(?<=Subject:).*\s?')
    try:
        from_msg = from_pattern.search(document).group()
    except AttributeError:
        from_msg = 'U'
        
    try:
        subject_msg = subject_pattern.search(document).group()
    except AttributeError:
        subject_msg = 'U'

    return from_msg, subject_msg


def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=3, ngram_range=(1, 2))


def extract_tokens(docs, tokenizer, filepath, filename):
    # read extracted tokens from file
    if os.path.isfile(os.path.join(filepath, filename)):
        with open(os.path.join(filepath, filename), "r") as tokens_file:
            return tokens_file.readlines()
    else:
        tokenized_list = []
        for text in docs:
            tokens = tokenizer(text)
            tokenized_list.append(tokens)
        with open(os.path.join(filepath, filename), "a") as tokens_file:
            for tokenized_text in tokenized_list:
                tokens_file.write(str(tokenized_text) + "\n")
    return tokenized_list


# calculate word vectors for document, append to list. When all vectors compiled, write it out into a separate file
def word_embeddings(spacy_nlp, docs, filepath, filename):
    """
    if os.path.isfile(os.path.join(filepath, filename)):
        compiled_vectors = np.load(os.path.join(filepath, filename), allow_pickle=True)
        return compiled_vectors
    else:
    """
        compiled_vectors = []
        with spacy_nlp.disable_pipes():
            for document in docs:
                document_vectors = np.array([token.vector for token in spacy_nlp(document)])
                compiled_vectors.append(document_vectors)
            compiled_vectors = np.array(compiled_vectors, dtype=object, copy=True, order='K', subok=True)
            np.save(os.path.join(filepath, filename), compiled_vectors)
        return compiled_vectors
