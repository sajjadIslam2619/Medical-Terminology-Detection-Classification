import os

MODEL_NAME = 'Bio_ClinicalBERT'
MODEL_ADDRESS = 'emilyalsentzer/Bio_ClinicalBERT'
MODEL_SAVED_DIR = 'trained_models'

RAW_TRAIN_DIRS = {
    'beth': '../Data/raw/concept_assertion_relation_training_data/beth',
    'partners': '../Data/raw/concept_assertion_relation_training_data/partners',
}

# development purpose small dataset
'''
RAW_TRAIN_DIRS = {
    'beth': './raw/concept_assertion_relation_training_data/beth',
    'partners': './raw/concept_assertion_relation_training_data/partners',
}
'''

TEST_DIR = '../Data/raw/reference_standard_for_test_data'
TEST_ASSERTION_PATH = '../Data/raw/reference_standard_for_test_data/ast'
TEST_TEXT_PATH = '../Data/raw/test_data'

# development purpose small dataset
'''
TEST_DIR = './raw/dev/reference_standard_for_test_data'
TEST_CONCEPT_PATH = './raw/dev/reference_standard_for_test_data/concepts'
TEST_ASSERTION_PATH = './raw/dev/reference_standard_for_test_data/ast'
TEST_TEXT_PATH = './raw/dev/test_data'
'''

DATA_SAVED_DIR = '../Data/processed/merged/'
DATA_REFORMAT_AST_SAVED_DIR = '../Data/processed/merged_assertion_reformat/'
DATA_REFORMAT_AST_LABEL_SAVED_DIR = '../Data/processed/merged/'

OUT_FILES = {
    'assertion_label_train': os.path.join(DATA_REFORMAT_AST_LABEL_SAVED_DIR, 'assertion_label_train.tsv'),
    'assertion_label_dev': os.path.join(DATA_REFORMAT_AST_LABEL_SAVED_DIR, 'assertion_label_dev.tsv'),
    'assertion_label_test': os.path.join(DATA_REFORMAT_AST_LABEL_SAVED_DIR, 'assertion_label_test.tsv'),
    'assertion_label_modified_train': os.path.join(DATA_REFORMAT_AST_LABEL_SAVED_DIR, 'assertion_label_modified_train'
                                                                                      '.tsv'),
    'assertion_label_modified_dev': os.path.join(DATA_REFORMAT_AST_LABEL_SAVED_DIR, 'assertion_label_modified_dev.tsv'),
    'assertion_label_modified_test': os.path.join(DATA_REFORMAT_AST_LABEL_SAVED_DIR,
                                                  'assertion_label_modified_test.tsv'),
    'label_assertion': '../label_vocab_assertion.txt',
    'label_ast': '../label_vocab_ast.txt'
}


