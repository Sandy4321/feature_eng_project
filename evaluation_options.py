# pd_series:
# Subject = text in Subject section of file
# pure_text = Only text in the body of the file
# text_data = Subject and pure_text merged together

# extensions:
# 1) subword_vectors: {pd_series}_sw_vect
# 2) doc_vectors: {pd_series}_m_vect
# 3) no_dup_words: {pd_series}_unq
# 4) lemmatized: {pd_series}_lem

# ordering:
# 1) BOW - lemmatize
# 2) subword_vectors - remove_dup -> subword_embedding
# 3) doc_vectors - remove_dup -> doc_mean_vectors

# vw model options
ngram_list = ['b2', 'b3', 'c2', 'c3']
vw_opts_list = [{
            # General options
            'random_seed': 7,
            # Output options
            'progress': 1000,
            # Update rule options
            'loss_function': 'logistic',
            # Weight options
            'bit_precision': 28,
            # Multiclass options
            'oaa': 20
        }]
for ngram in ngram_list:
    vw_opts_list.append(
        {
            # General options
            'random_seed': 7,
            # Output options
            'progress': 1000,
            # Example Manipulation options
            'ngram': ngram,
            # Update rule options
            'loss_function': 'logistic',
            # Weight options
            'bit_precision': 28,
            # Multiclass options
            'oaa': 20
        }
    )
    # vw_opts_list
    # [0] - no ngrams
    # [1] - pure_text 2gram
    # [2] - pure_text 3gram
    # [3] - text_data 2gram
    # [4] - text_data 3gram

combinations_to_test = [
    # only pure_text
    [['pure_text_lem'],
     [vw_opts_list[0], vw_opts_list[1], vw_opts_list[2]]],  # pure BOW
    [['pure_text_unq_sw_vect'],
     [vw_opts_list[0]]],  # pure subword_vectors
    [['pure_text_lem', 'pure_text_unq_sw_vect'],
     [vw_opts_list[0], vw_opts_list[1], vw_opts_list[2]]],  # BOW and subword_vectors

    # only text_data
    [['text_data_lem'],
     [vw_opts_list[0], vw_opts_list[3], vw_opts_list[4]]],  # pure BOW
    [['text_data_unq_sw_vect'],
     [vw_opts_list[0]]],  # pure subword_vectors
    [['text_data_lem', 'text_data_unq_sw_vect'],
     [vw_opts_list[0], vw_opts_list[3], vw_opts_list[4]]],  # BOW and subword_vectors

    # pure_text and Subject
    [['pure_text_lem', 'Subject_lem'],
     [vw_opts_list[0], vw_opts_list[1], vw_opts_list[2]]],  # pure BOW
    [['pure_text_unq_sw_vect', 'Subject_unq_sw_vect'],
     [vw_opts_list[0]]],  # pure subword_vectors
    [['pure_text_lem', 'Subject_lem', 'pure_text_unq_sw_vect', 'Subject_unq_sw_vect'],
     [vw_opts_list[0], vw_opts_list[1], vw_opts_list[2]]],  # BOW and subword_vectors
    [['pure_text_unq_sw_vect', 'Subject_unq_m_vect'],
     [vw_opts_list[0]]],  # subword_vectors and doc_vectors
    [['pure_text_lem', 'Subject_lem', 'pure_text_unq_sw_vect', 'Subject_unq_m_vect'],
     [vw_opts_list[0], vw_opts_list[1], vw_opts_list[2]]],  # BOW, subword_vectors and doc_vectors
    [['pure_text_lem', 'Subject_lem', 'pure_text_unq_sw_vect'],
     [vw_opts_list[0], vw_opts_list[1], vw_opts_list[2]]],  # BOW and subword_vectors
]

