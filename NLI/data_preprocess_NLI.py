import os
import sys
import inspect

# In VS-code to import Assertion module it is needed to add parent folder directory in sys.path 
# In pyCharm only import works because NLI is a python package not just a folder. 
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from Assertion.util import config


import pandas as pd
import re
from random import seed
from random import randint
import numpy as np

label_names = ['present', 'absent', 'possible', 'conditional', 'hypothetical', 'associated_with_someone_else']


def add_hypothesis_col (df):

    df["hypothesis"] = pd.NaT
    df_generate_data = pd.DataFrame()
    df_generate_data["label"] = pd.NaT 
    df_generate_data["premise"] = pd.NaT 
    df_generate_data["hypothesis"] = pd.NaT

    for i in df.index:
        sentence = df['sentence'][i]
        label = df['label'][i]
        label_index = label_names.index(label)
        pos_arr = []
        cnt = 0
        for match in re.finditer(r"\[entity\]", sentence, re.IGNORECASE):
            cnt += 1
            #print(cnt, "st match start index", match.start(), "End index", match.end())
            pos_arr.append(match.start())
            pos_arr.append( match.end())
        
        entity = sentence[pos_arr[1]: pos_arr[2]]
        hypothesis = ''
        hypothesis = entity.strip() + ' is ' + label
        df ['hypothesis'][i] = hypothesis
        df ['label'][i] = 'entailment'

        seed(1)
        rand_value_list = []
        max_no_of_generate_hypothesis = 0
        for _ in range (10):
            rand_value = randint(0, 5)
            if rand_value == label_index: continue # this label is entailment, previously added in data-frame
            elif rand_value in rand_value_list: continue # this label is already added in non-entailment data-frame
            elif max_no_of_generate_hypothesis == 5: break # maximum no of generated hypothesis, will not be more than 5
            rand_value_list.append(rand_value)
            max_no_of_generate_hypothesis = max_no_of_generate_hypothesis + 1

            generated_label = 'contradiction'
            generate_hypothesis = entity.strip() + ' is ' + label_names[rand_value]
            df_generate_data.loc[len(df_generate_data.index)] = [generated_label, sentence, generate_hypothesis]

    df = df.rename(columns={'sentence': 'premise'})
    df = df[['label', 'premise', 'hypothesis']]

    df =  pd.concat([df, df_generate_data], axis=0)

    return df

def add_hypothesis_col_test (df):

    df["hypothesis"] = pd.NaT
    df_generate_data = pd.DataFrame()
    df_generate_data["label"] = pd.NaT 
    df_generate_data["premise"] = pd.NaT 
    df_generate_data["hypothesis"] = pd.NaT

    for i in df.index:
        sentence = df['sentence'][i]
        label = df['label'][i]
        label_index = label_names.index(label)
        pos_arr = []
        cnt = 0
        for match in re.finditer(r"\[entity\]", sentence, re.IGNORECASE):
            cnt += 1
            #print(cnt, "st match start index", match.start(), "End index", match.end())
            pos_arr.append(match.start())
            pos_arr.append( match.end())
        
        entity = sentence[pos_arr[1]: pos_arr[2]]
        hypothesis = ''
        hypothesis = entity.strip() + ' is ' + label
        df ['label'][i] = 'entailment'
        df ['hypothesis'][i] = hypothesis

        seed(1)
        rand_value_list = []
        max_no_of_generate_hypothesis = 0
        for _ in range (10):
            rand_value = randint(0, 5)
            if rand_value == label_index: continue # this label is entailment, previously added in data-frame
            elif rand_value in rand_value_list: continue # this label is already added in non-entailment data-frame
            elif max_no_of_generate_hypothesis == 5: break # maximum no of generated hypothesis, will not be more than 5
            rand_value_list.append(rand_value)
            max_no_of_generate_hypothesis = max_no_of_generate_hypothesis + 1

            generated_label = 'contradiction'
            generate_hypothesis = entity.strip() + ' is ' + label_names[rand_value]
            df_generate_data.loc[len(df_generate_data.index)] = [generated_label, sentence, generate_hypothesis]

    df = df.rename(columns={'sentence': 'premise'})
    df = df[['label', 'premise', 'hypothesis']]

    df =  pd.concat([df, df_generate_data], axis=0)

    return df

def sentence_length(sentence):
    length = len(sentence)
    if length < 200: 
        return sentence
    else: np.NaN

def main():

    train_df = pd.read_csv(config.OUT_FILES['assertion_6_label_modified_train'], sep='\t', header=0)
    dev_df = pd.read_csv(config.OUT_FILES['assertion_6_label_modified_dev'], sep='\t', header=0)
    test_df = pd.read_csv(config.OUT_FILES['assertion_6_label_modified_test'], sep='\t', header=0)
    test_df = test_df.drop(test_df.index[[11019, 7045, 6835, 4458]])
    test_df['sentence'] = test_df['sentence'].apply(lambda x : sentence_length(x))
    test_df = test_df.dropna()

    print('Train data (Original label counts): ', len(train_df))
    print(train_df['label'].value_counts())
    print('---------------------------------')
    print('Dev data (original label counts): ', len(dev_df))
    print(dev_df['label'].value_counts())
    print('---------------------------------')
    print('Test data (Original label counts): ', len(test_df))
    print(test_df['label'].value_counts())
    

    train_df = add_hypothesis_col(train_df)
    dev_df = add_hypothesis_col(dev_df)
    test_df = add_hypothesis_col_test(test_df)

    #train_df = train_df.sample(15000)    
    #dev_df = dev_df.sample(1416)
    #test_df = test_df.sample(12225)
    #print(len(train_df))
    #print(train_df.sample(10))
    #print(dev_df.head(2))
    #print(test_df.head(2))

    
    print('Train data (NLI label counts): ', len(train_df))
    print(train_df['label'].value_counts())
    print('---------------------------------')
    print('Dev data (NLI label counts): ', len(dev_df))
    print(dev_df['label'].value_counts())
    print('---------------------------------')
    print('Test data (NLI label counts): ', len(test_df))
    print(test_df['label'].value_counts())
    
    train_df.to_csv(config.OUT_FILES['assertion_NLI_train'], sep="\t")
    dev_df.to_csv(config.OUT_FILES['assertion_NLI_dev'], sep="\t")
    test_df.to_csv(config.OUT_FILES['assertion_NLI_test'], sep="\t")
    

if __name__ == "__main__":
    main()