# general python modules
import os
import re
import pandas as pd
from tqdm import tqdm
# project modules
from data_wrangling import load_from_pickle


def sk_to_pd_df(sk_bunch, target_dict):
    # converting from sk utils Bunch to Pandas dataframe
    pd_df = pd.DataFrame(data=[sk_bunch.data, sk_bunch.target])
    pd_df = pd_df.transpose()
    pd_df.columns = ['text', 'target']
    pd_df["target"] = pd_df["target"].map(target_dict)
    return pd_df


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


def extract_text_info(pd_df):
    text_content_list = []
    email_subject_rgx = re.compile(r'(?<=Subject: ).*')
    subject_content_rgx = re.compile(r'(?<=[rR]e: ).*')
    metadata_rgx_1 = re.compile(r'From:.*\nSubject: .*')
    metadata_rgx_2 = re.compile(r'Subject: .*\nFrom: .*')
    for doc in tqdm(pd_df['text']):
        subject = email_subject_rgx.search(doc).group()
        try:
            subject_content = subject_content_rgx.search(subject).group()
        except AttributeError:  # no '(R/r)e: ' in subject
            subject_content = subject
        try:
            metadata = metadata_rgx_1.search(doc).group()
        except AttributeError:
            try:
                metadata = metadata_rgx_2.search(doc).group()
            except AttributeError:
                print("AttributeError: Check df['text'], index= ")
        try:
            text_content = f"{subject_content} {doc.replace(metadata, '')}"
            text_content_list.append(text_content)
        except NameError as error:
            print(error)
    pd_df.insert(loc=len(pd_df.columns), column='text_data', value=text_content_list)

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

            if len(tokens_no_sw_string) == 0:
                tokens_no_sw_string = ' '
            pd_df.loc[:, pd_series].replace(to_replace=doc, value=tokens_no_sw_string, inplace=True)

    return 0


def text_pre_processing(spacy_nlp, pd_df, filepath, filename, load_file=False):
    # generate new dataframe
    if not load_file:
        print('Extracting pre-processed text...')
        # remove pre-existing dataframe
        if os.path.isfile(os.path.join(filepath, filename)):
            os.remove(os.path.join(filepath, filename))
        extract_section_Subject(pd_df)
        extract_section_pure_text(pd_df)
        extract_text_info(pd_df)
        text_rmv_noise(spacy_nlp, pd_df, 'Subject')
        text_rmv_noise(spacy_nlp, pd_df, 'pure_text')
        text_rmv_noise(spacy_nlp, pd_df, 'text_data')
        pd_df.drop(columns=['text'], inplace=True)

        # store pre-processed text into a separate file for later retrieval
        storage_f = os.path.join(filepath, filename)
        pd_df.to_pickle(path=storage_f)
        print('Pre-processed text stored!')
        print('Pre-processed text loaded!')

    # load dataframe from memory
    if load_file:
        print('Loading pre-processed text...')
        pd_df = load_from_pickle(filepath, filename)
        print('Pre-processed text loaded!')

    return pd_df
