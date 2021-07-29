v2.2:
Changes:
- Integrated hyperopt to optimize l1 and l2 regularization

main.py:
- Additional use_hyperopt arg to choose to run with hyperopt or not

evaluation_options:
- 2 different combinations_to_test based on use_hyperopt arg

model_train_eval:
- store_results_hyperopt and model_train_test_hyperopt to support running with hyperopt
- store_results now saves f1_score as well

v2.1:
Changes:
- Added support for ngrams with vowpal wabbit

convert_to_vw_format:
- Added unique alphabetical character at start of each namespace

evaluation_options:
- combinations_to_test
- old format: [[feature_1, feature_2, ...], [feature_1, ...], ...]
- new format: [[['feature_1', ...], [vw_opt1, ...]], [['feature_1', ...], [vw_opt1, ...]]...]

New highest accuracy score = 0.9203314212874443
Features: text_data_lem (BOW)
vw_opts: 'ngram': c3 (text_data_lem 3-grams)
v2.0:

main (2 new args):
- process_data <- Whether or not to process data and store it
- train_test_model <- Whether or not to train and test the vw_model

data_pre_processing:
- Extracts Subject, pure_text and text_info sections

data_wrangling:
- Lemmatizes using spacy
- Remove duplicate words using spacy
- Find doc_vectors using bpemb_en
- Find subword_vect using bpemb_en

model_train_eval:
- Trains a vw_model using the features extracted from data_wrangling
- Evaluates the vw_model and saves the classification report in a .csv file, combines features and accuracy score in a .txt file

evaluation_options:
- Stores which features to use for vw_model training/evaluation
- Stores vw_model config

Dependencies downloaded separately:
- bpemb_en model (https://bpemb.h-its.org/en/en.wiki.bpe.vs200000.model)
- bpemb_en vocab (https://bpemb.h-its.org/en/en.wiki.bpe.vs200000.d300.w2v.bin.tar.gz)
- save bpemb_en model and vocab in \project_working_directory\bpemb_model
- spacy (python -m spacy download en_core_web_md)

Removed:

data_wrangling:
- original baseline bag_of_words fn
- word_embedding using spacy (replaced with subword_vect using bpemb_en)
- load_from_csv (replaced with load_from_pickle)

Features Evaluation:
- Highest accuracy score = 0.9162948799660081
- Features: text_data and Subject (BOW and subword_vectors)

Accuracy score | Features
1) pure_text
- 0.8742298704057786 | pure_text_lem (pure BOW)
- 0.8863394943700871 | pure_text_unq_sw_vect (pure subword_vectors)
- 0.8859145952836202 | pure_text_lem, pure_text_unq_sw_vect  (BOW and subword_vectors)

2) text_data
- 0.9029105587422988 | text_data_lem (pure BOW)
- 0.9012109623964308 | text_data_unq_sw_vect (pure subword_vectors)
- 0.9009985128531973 | text_data_lem, text_data_unq_sw_vect (BOW and subword_vectors)

3) text_data and Subject
- 0.9003611642234969 | pure_text_lem, Subject_lem (pure BOW)
- 0.9158699808795411 | pure_text_unq_sw_vect, Subject_unq_sw_vect (pure subword_vectors)
- 0.9162948799660081 | pure_text_lem, Subject_lem, pure_text_unq_sw_vect, Subject_unq_sw_vect (BOW and subword_vectors)
- 0.8869768429997875 | pure_text_unq_sw_vect, Subject_unq_m_vect (subword_vectors and doc_vectors)
- 0.8878266411727215 | pure_text_lem, Subject_lem, pure_text_unq_sw_vect, Subject_unq_m_vect (BOW, subword_vectors and doc_vectors)

v1.0:

highest accuracy score achieved is 0.910558742298704 (BOW From, pure_text, doc.vector Subject)

main:
- From - processed as text
- Subject - processed as doc.vector
- pure_text - processed as token.vector

data_pre_processing:
- Extracts From, Subject and pure_text sections

data_wrangling:
- lemmatize for BOW
- extract token.vector
- extract doc.vector

convert_to_vw_format:
- Converts text, word.vectors and doc.vectors into vw format for modelling

vw_config:
- project constants
- vw_model options