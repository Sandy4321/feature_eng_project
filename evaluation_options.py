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


combinations_to_test = [
    # only pure_text
    ['pure_text_lem'],  # pure BOW
    ['pure_text_unq_sw_vect'],  # pure subword_vectors
    ['pure_text_lem', 'pure_text_unq_sw_vect'],  # BOW and subword_vectors

    # only text_data
    ['text_data_lem'],  # pure BOW
    ['text_data_unq_sw_vect'],  # pure subword_vectors
    ['text_data_lem', 'text_data_unq_sw_vect'],  # BOW and subword_vectors

    # text_data and Subject
    ['pure_text_lem', 'Subject_lem'],  # pure BOW
    ['pure_text_unq_sw_vect', 'Subject_unq_sw_vect'],  # pure subword_vectors
    ['pure_text_unq_sw_vect', 'Subject_unq_m_vect'],  # subword_vectors and doc_vectors,
    ['pure_text_lem', 'Subject_lem', 'pure_text_unq_sw_vect', 'Subject_unq_sw_vect'],  # BOW and subword_vectors
    ['pure_text_lem', 'Subject_lem', 'pure_text_unq_sw_vect', 'Subject_unq_m_vect']  # BOW, subword_vectors and doc_vectors
]


# vw model options
random_state_list = [7]
vw_opts_list = []
for random_state in random_state_list:
    vw_opts_list.append(
        {
            # General options
            "random_seed": random_state,  # changing random_seed to account for anomalous behaviour
            # Output options
            "progress": 1000,
            # Update rule options
            "loss_function": "logistic",
            # Weight options
            "bit_precision": 28,
            # Multiclass options
            "oaa": 20
        }
    )
