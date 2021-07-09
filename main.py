# NLP modules
import gc

import spacy
# ML modules
from vowpalwabbit import pyvw
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from bpemb import BPEmb
# project modules
from project_constants import (
    target_dict,
    raw_data_fpath,
    bpemb_model_fpath,
    processed_data_fpath,
    pre_processed_filename,
    dups_removed_filename,
    pure_text_lemmatized_filename,
    words_vectorized_filename,
    doc_vectorized_filename,
    pred_fpath,
    pred_filename
)
from data_wrangling import text_lemmatization, doc_mean_vectors, remove_dup_words, subword_embedding
from data_pre_processing import sk_to_pd_df, text_pre_processing
from convert_to_vw_format import pd_to_vw_fmt
from evaluation_options import combinations_to_test
from model_train_eval import to_vw, train_model, eval_model


def main(process_data, train_test_model, run=1):
    # process data for training and testing of vw_model
    if process_data:

        #  loading in data
        newsgroups = load_files(raw_data_fpath, encoding='ANSI')
        df = sk_to_pd_df(newsgroups, target_dict)

        # data processing
        spacy_nlp = spacy.load('en_core_web_md')
        bpemb_en = BPEmb(model_file=f"{bpemb_model_fpath}\\en.wiki.bpe.vs200000.model",
                         emb_file=f"{bpemb_model_fpath}\\en.wiki.bpe.vs200000.d300.w2v.bin", dim=300)

        # base pd_df sections: target | Subject | pure_text | text_data
        df = text_pre_processing(spacy_nlp, df, processed_data_fpath, pre_processed_filename,
                                 load_file=True)

        # for BOW features
        # current pd_df sections: target | Subject | pure_text | text_data
        #                                | Subject_lem | pure_text_lem | text_data_lem
        df = text_lemmatization(spacy_nlp, df, ['Subject', 'pure_text', 'text_data'],
                                processed_data_fpath, pure_text_lemmatized_filename,
                                load_file=True)

        # creates new pd_series with key f"{pd_series}_unq"
        # for subword_vectors
        # current pd_df sections: target | Subject | pure_text | text_data
        #                                | Subject_lem | pure_text_lem | text_data_lem
        #                                | Subject_unq | pure_text_unq | text_data_unq
        df = remove_dup_words(spacy_nlp, df, ['Subject', 'pure_text', 'text_data'],
                              processed_data_fpath, dups_removed_filename, load_file=True)

        # creates new pd_series with key f"{pd_series}_m_vect"
        # current pd_df sections: target | Subject | pure_text | text_data
        #                                | Subject_lem | pure_text_lem | text_data_lem
        #                                | Subject_unq | pure_text_unq | text_data_unq
        #                                | Subject_unq_m_vect
        df = doc_mean_vectors(bpemb_en, df, ['Subject_unq'], processed_data_fpath, doc_vectorized_filename,
                              load_file=True)

        # creates new pd_series with key f"{pd_series}_sw_vect"
        # current pd_df sections: target | Subject | pure_text | text_data
        #                                | Subject_lem | pure_text_lem | text_data_lem
        #                                | Subject_unq | pure_text_unq | text_data_unq
        #                                | Subject_unq_m_vect
        #                                | Subject_unq_sw_vect | pure_text_unq_sw_vect | text_data_unq_sw_vect
        df = subword_embedding(bpemb_en, df, ['Subject_unq', 'pure_text_unq', 'text_data_unq'],
                               processed_data_fpath, words_vectorized_filename, load_file=True)

        # Each series that is converted in VW format have a unique first char for namespace
        # store training data
        data_train, data_test = train_test_split(df, test_size=0.25, random_state=1)
        pd_to_vw_fmt(pd_df=data_train,
                     text_sections_to_convert=['Subject_lem',
                                               'pure_text_lem',
                                               'text_data_lem'],
                     doc_vector_sections_to_convert=['Subject_unq_m_vect'],
                     subword_vector_sections_to_convert=['Subject_unq_sw_vect',
                                                         'pure_text_unq_sw_vect',
                                                         'text_data_unq_sw_vect'],
                     filepath=processed_data_fpath, train=True,)
        # store testing data
        pd_to_vw_fmt(pd_df=data_test,
                     text_sections_to_convert=['Subject_lem',
                                               'pure_text_lem',
                                               'text_data_lem'],
                     doc_vector_sections_to_convert=['Subject_unq_m_vect'],
                     subword_vector_sections_to_convert=['Subject_unq_sw_vect',
                                                         'pure_text_unq_sw_vect',
                                                         'text_data_unq_sw_vect'],
                     filepath=processed_data_fpath, train=False)

    # run each combination of features through VW
    # save classification report as .csv file
    # save features and accuracy score together in a .txt file
    if train_test_model:
        for features in combinations_to_test:
            train_str_list = to_vw(processed_data_fpath, features[0], train=True)
            test_str_list = to_vw(processed_data_fpath, features[0], train=False)
            for vw_opts in features[1]:
                print(vw_opts)
                try:
                    ngram = vw_opts['ngram'][-1]
                except KeyError:
                    ngram = 0
                # generate vw_model with corresponding params
                vw_model = pyvw.vw(**vw_opts)

                # train vw_model
                train_model(vw_model, train_str_list)

                # evaluate vw_model against test set
                eval_model(vw_model, test_str_list, processed_data_fpath,
                           features[0], pred_fpath, pred_filename,
                           ngram, run=run)
            # to prevent MemoryError
            del train_str_list
            del test_str_list
            gc.collect()
            run += 1

    return 0


if __name__ == "__main__":
    main(process_data=False, train_test_model=True, run=1)
