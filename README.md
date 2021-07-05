# feature_eng_project
v1.1:
highest accuracy score achieved is 0.9162948799660081
(Features: pure_text_lem, Subject_lem, pure_text_unq_sw_vect, Subject_unq_sw_vect)

main (2 new args):
- process_data <- Whether or not to process data and store it
- train_test_model <- Whether or not to train and test the vw_model

data_pre_processing:
- Extracts Subject, pure_text and text_info sections

data_wrangling:
- Lemmatizes using spacy
- Remove duplicate words
- Find doc vectors using spacy
- Find subword_vect using bpemb_en

model_train_eval:
- Trains a vw_model using the features extracted from data_wrangling
- Evaluates the vw_model and saves the classification report in a .csv file, combines features and accuracy score in a .txt file

evaluation_options:
- Stores which features to use for vw_model training/evaluation
- Stores vw_model config

v1.0:
highest accuracy score achieved is 0.910558742298704 (BOW From, pure_text, doc.vector Subject)
main:
From - processed as text
Subject - processed as doc.vector
pure_text - processed as token.vector

data_pre_processing:
- Extracts From, Subject and pure_text sections

data_wrangling:
- lemmatize for BOW
- extract token.vector
- extract doc.vector

convert_to_vw_format:
Converts text, word.vectors and doc.vectors into vw format for modelling

vw_config:
- project constants
- vw_model options