# automating testing WIP
# pd_series: 'From', 'Subject', 'pure_text'
# extensions:
# 1) word_vectors: {pd_series}_w_vect
# 2) doc_vectors: {pd_series}_m_vect

# run each combination of features through VW, save classification report and accuracy score in a separate file
combinations_to_test = [
    {
        'text': [''],
        'word_vectors': [''],
        'doc_vectors': ['']
    }
]

# vw model options
vw_opts = {
    # General options
    "random_seed": 1,
    # Input options
    # Output options
    "progress": 1000,
    # Example Manipulation options
    # Update rule options
    "loss_function": "logistic",
    # Weight options
    "bit_precision": 28,
    # Holdout options
    # Feature namespace options
    # Multiclass options
    "oaa": 20
    # Other options
}

random_states = [7, 11, 54]  # for replicability, numbers were randomly chosen between 1 and 100
