import os
proj_filepath = os.getcwd()
raw_data_fpath = os.path.join(proj_filepath, '20news_noDup')
bpemb_model_fpath = os.path.join(proj_filepath, 'bpemb_model')

processed_data_fpath = os.path.join(proj_filepath, 'processed_data')
pre_processed_filename = 'pre_processed_text'
dups_removed_filename = 'no_dup_words'
pure_text_lemmatized_filename = 'pure_text_lemmatized'
words_vectorized_filename = 'data_word_vectors'
doc_vectorized_filename = 'data_doc_vectors'

pred_fpath = os.path.join(proj_filepath, 'predictions')
pred_filename = 'pred_vw'

# for conversion to vw format for multiclass model
target_dict = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 6,
        6: 7,
        7: 8,
        8: 9,
        9: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 14,
        14: 15,
        15: 16,
        16: 17,
        17: 18,
        18: 19,
        19: 20
    }