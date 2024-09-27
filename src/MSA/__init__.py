## use __init__.py file to put global parameters for use.
col_time = 'date'
col_seq = 'event'
date_format = '%Y-%m-%d'
max_transposition_date = 10


## levenshtein.py contains the core function to calculate similarity of two given sequences
## low_utilities.py are the functions with no dependancy requirement
## utilities.py are the functions with dependancy requirement like pandas
## MSA.py & Pre_MSA.py are standalone scripts
## visual_tools.py are the function with a lot plot library for visual
# objects.py for class like substitue_matrix and patient. 