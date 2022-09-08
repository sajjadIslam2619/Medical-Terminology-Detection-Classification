from util import config
import pandas as pd
import re


def add_hypothesis_col (df):

    df["hypothesis"] = pd.NaT

    for i in df.index:
        sentence = df['sentence'][i]
        label = df['label'][i]
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

    df = df.rename(columns={'sentence': 'premise'})
    df = df[['label', 'premise', 'hypothesis']]

    return df

def add_hypothesis_col_test (df):

    df["hypothesis"] = pd.NaT

    for i in df.index:
        sentence = df['sentence'][i]
        label = df['label'][i]
        pos_arr = []
        cnt = 0
        for match in re.finditer(r"\[entity\]", sentence, re.IGNORECASE):
            cnt += 1
            #print(cnt, "st match start index", match.start(), "End index", match.end())
            pos_arr.append(match.start())
            pos_arr.append( match.end())
        
        entity = sentence[pos_arr[1]: pos_arr[2]]
        hypothesis = ''
        hypothesis = entity.strip() + ' is ' + 'entity'
        df ['hypothesis'][i] = hypothesis

    df = df.rename(columns={'sentence': 'premise'})
    df = df[['label', 'premise', 'hypothesis']]

    return df

def main():
    #print('PATH ::: ', config.OUT_FILES['assertion_label_dev'])

    train_df = pd.read_csv(config.OUT_FILES['assertion_label_train'], sep='\t', header=0)
    dev_df = pd.read_csv(config.OUT_FILES['assertion_label_dev'], sep='\t', header=0)
    test_df = pd.read_csv(config.OUT_FILES['assertion_label_test'], sep='\t', header=0)
    

    '''
    print('Train data (Original label counts)')
    print(train_df['label'].value_counts())
    print('---------------------------------')
    print('Dev data (original label counts)')
    print(dev_df['label'].value_counts())
    print('---------------------------------')
    print('Test data (Original label counts)')
    print(test_df['label'].value_counts())
    '''

    train_df = add_hypothesis_col(train_df)
    dev_df = add_hypothesis_col(dev_df)
    test_df = add_hypothesis_col_test(test_df)

    #print(train_df.head(2))
    #print(dev_df.head(2))
    #print(test_df.head(2))

    '''
    print('Train data (NLI label counts)')
    print(train_df['label'].value_counts())
    print('---------------------------------')
    print('Dev data (NLI label counts)')
    print(dev_df['label'].value_counts())
    print('---------------------------------')
    print('Test data (NLI label counts)')
    print(test_df['label'].value_counts())
    '''
    train_df.to_csv(config.OUT_FILES['assertion_NLI_train'], sep="\t")
    dev_df.to_csv(config.OUT_FILES['assertion_NLI_dev'], sep="\t")
    test_df.to_csv(config.OUT_FILES['assertion_NLI_test'], sep="\t")
    

if __name__ == "__main__":
    main()