import os
import subprocess
import torch


MAX_LEN = 64
BATCH_NUM = 32
EPOCH = 10
FULL_FINETUNING = True # True: fine tuning all the layers  False: only fine tuning the classifier layers
LEARNING_RATE = 3e-5
MODEL_NAME = 'Bio_ClinicalBERT'
MODEL_ADDRESS = 'emilyalsentzer/Bio_ClinicalBERT'
MODEL_SAVED_DIR = './trained_models'

RAW_TRAIN_DIRS = {
    'beth': './Data/raw/concept_assertion_relation_training_data/beth',
    'partners': './Data/raw/concept_assertion_relation_training_data/partners',
}
TEST_DIR = './Data/raw/reference_standard_for_test_data'
TEST_CONCEPT_PATH = './Data/raw/reference_standard_for_test_data/concepts'
TEST_TEXT_PATH = './Data/raw/test_data'


DATA_SAVED_DIR = './Data/processed/merged/'
INDIVIDUAL_TEST = './Data/processed/test/'
OUT_FILES = {
    'merged_train': os.path.join(DATA_SAVED_DIR, 'train.tsv'),
    'merged_dev':   os.path.join(DATA_SAVED_DIR, 'dev.tsv'),
    'merged_test':  os.path.join(DATA_SAVED_DIR, 'test.tsv'),
    'label': './Data/label_vocab.txt'
}

data_path_train = OUT_FILES['merged_train']
data_path_dev = OUT_FILES['merged_dev']
data_path_test = OUT_FILES['merged_test']
label_path = OUT_FILES['label']
# vocabulary_path = "./Bio_ClinicalBERT/vocab.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add multi GPU support
multi_gpus = torch.cuda.device_count() > 1
if not os.path.exists(DATA_SAVED_DIR):
    os.makedirs(DATA_SAVED_DIR)
if not os.path.exists(INDIVIDUAL_TEST):
    os.makedirs(INDIVIDUAL_TEST)
if not os.path.exists(MODEL_SAVED_DIR):
    os.makedirs(MODEL_SAVED_DIR)
if not os.path.exists(MODEL_NAME):
    subprocess.Popen(['git', 'clone', 'https://huggingface.co/'+MODEL_ADDRESS])


def main():
    print('debug code')
    print(device)
    if not os.path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)
    # with open(OUT_FILES['label']) as f: l = [i.strip('\n') for i in f.readlines()]
    # print(l)
    
    

if __name__ == "__main__":
    main()