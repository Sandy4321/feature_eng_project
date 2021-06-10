# general python modules
import os
import re
import pandas as pd
from tqdm import tqdm
# project modules
from data_wrangling import load_from_csv


def sk_to_pd_df(sk_bunch, target_dict):
    # converting from sk utils Bunch to Pandas dataframe
    pd_df = pd.DataFrame(data=[sk_bunch.data, sk_bunch.target])
    pd_df = pd_df.transpose()
    pd_df.columns = ['text', 'target']
    pd_df['text'].astype("string")

    pd_df["target"] = pd_df["target"].map(target_dict)
    return pd_df


# process From section, replace all non-alphanumeric with space
# currently known formats:
# abc@a.b.c
# abc@a.b.c (abc)
# (abc)
def extract_section_From(pd_df):
    email_sender_rgx = re.compile(r'(?<=From: ).*')
    non_alphanumeric_rgx = re.compile(r'[\W_]')
    multiple_spaces_rgx = re.compile(r'  +')
    # extract all text after 'From: ' and before newline
    pd_df.insert(loc=len(pd_df.columns), column="From", value=pd_df["text"].apply(lambda text: email_sender_rgx.search(text).group()))
    # match all non-alphanumeric characters and replace them with spaces
    pd_df['From'] = pd_df['From'].str.replace(pat=non_alphanumeric_rgx, repl=' ')
    # replace multiple spaces between words with only 1 space
    pd_df['From'] = pd_df['From'].str.replace(pat=multiple_spaces_rgx, repl=' ')
    return 0


# process Subject section
# remove (R/r)e:
def extract_section_Subject(pd_df):
    email_subject_rgx = re.compile(r'(?<=Subject: ).*')
    subject_content_rgx = re.compile(r'(?<=[rR]e: ).*')
    subject_content_dict = {}
    pd_df.insert(loc=len(pd_df.columns), column="Subject",
                 value=pd_df["text"].apply(lambda text: email_subject_rgx.search(text).group()))
    for doc in tqdm(pd_df['Subject']):
        try:
            subject_content = subject_content_rgx.search(doc).group()
            subject_content_dict[doc] = subject_content
        except AttributeError:  # no '(R/r)e: ' in subject
            subject_content_dict[doc] = doc
    pd_df['Subject'] = pd_df["Subject"].map(subject_content_dict)
    print(len(subject_content_dict))

    return pd_df


def extract_section_pure_text(pd_df):
    metadata_rgx_1 = re.compile(r'From:.*\nSubject: .*')
    metadata_rgx_2 = re.compile(r'Subject: .*\nFrom: .*')
    ctr2 = 0
    pure_text_list = []
    for doc in tqdm(pd_df['text']):
        try:
            metadata = metadata_rgx_1.search(doc).group()
        except AttributeError:
            try:
                metadata = metadata_rgx_2.search(doc).group()
            except AttributeError:
                print("AttributeError: Check df['text'], index= ", ctr2)
        try:
            pure_text_list.append(doc.replace(metadata, ''))
        except NameError as error:
            print(error)
        ctr2 += 1
    pd_df.insert(loc=len(pd_df.columns), column="pure_text", value=pure_text_list)

    return 0


def text_rmv_noise(spacy_nlp, pd_df, pd_series):
    # tokenize text
    with spacy_nlp.select_pipes(enable='tokenizer'):
        for doc in tqdm(pd_df[pd_series]):
            tokens = spacy_nlp(doc)
            tokens_no_sw = []
            # keep only alphabets, drop everything else
            # drop stop words
            # return lowercase form of text
            for token in tokens:
                if token.is_alpha:
                    if not token.is_stop:
                        tokens_no_sw.append(token.lower_)

            # converting from list to string
            separator = ' '
            tokens_no_sw_string = separator.join(tokens_no_sw)
            # blank line
            if len(tokens_no_sw_string) == 0:
                tokens_no_sw_string = ' '
            pd_df[pd_series].replace(to_replace=doc, value=tokens_no_sw_string, inplace=True)

    return 0


def text_pre_processing(spacy_nlp, pd_df, pd_series_subject, pd_series_text, filepath, filename, load_file=False):
    # generate new dataframe
    if not load_file:
        # remove pre-existing dataframe
        if os.path.isfile(os.path.join(filepath, filename)):
            os.remove(os.path.join(filepath, filename))
        extract_section_From(pd_df)
        extract_section_Subject(pd_df)
        extract_section_pure_text(pd_df)
        text_rmv_noise(spacy_nlp, pd_df, pd_series_subject)
        text_rmv_noise(spacy_nlp, pd_df, pd_series_text)
        pd_df.drop(columns=['text'], inplace=True)
        # store pre-processed text into a separate file for later retrieval
        with open(os.path.join(filepath, filename), 'w') as storage_f:
            pd_df.to_csv(path_or_buf=storage_f, index=False)

    # load dataframe from memory
    if load_file:
        pd_df = load_from_csv(filepath, filename)

    return pd_df
