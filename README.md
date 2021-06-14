# feature_eng_project

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