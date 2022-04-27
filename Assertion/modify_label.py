from util import config
import pandas as pd


def replace_ast_label(ast_label):
    if ast_label == 'present':
        return 'present'
    elif ast_label == 'O':
        return 'O'
    else:
        return 'not-present'
    return ''


def replace_ast_3_label(ast_label):
    if ast_label == 'present':
        return 'present'
    elif ast_label == 'possible':
        return 'possible'
    else:
        return 'not-present'
    return ''


def main():
    dev_df = pd.read_csv(config.OUT_FILES['assertion_label_dev'], sep='\t', header=0)
    test_df = pd.read_csv(config.OUT_FILES['assertion_label_test'], sep='\t', header=0)
    train_df = pd.read_csv(config.OUT_FILES['assertion_label_train'], sep='\t', header=0)

    print('Train data (Original label counts)')
    print(train_df['label'].value_counts())
    print('---------------------------------')
    print('Dev data (original label counts)')
    print(dev_df['label'].value_counts())
    print('---------------------------------')
    print('Test data (Original label counts)')
    print(test_df['label'].value_counts())

    '''
    dev_df['label'] = dev_df['label'].apply(lambda ast_label: replace_ast_label(ast_label))
    test_df['label'] = test_df['label'].apply(lambda ast_label: replace_ast_label(ast_label))
    train_df['label'] = train_df['label'].apply(lambda ast_label: replace_ast_label(ast_label))
    
    print('Dev data (modified label counts)')
    print(dev_df['label'].value_counts())
    print('---------------------------------')
    print('Test data (modified label counts)')
    print(test_df['label'].value_counts())
    print('---------------------------------')
    print('Train data (modified label counts)')
    print(train_df['label'].value_counts())
    
    dev_df.to_csv(config.OUT_FILES['assertion_label_modified_dev'], sep="\t")
    test_df.to_csv(config.OUT_FILES['assertion_label_modified_test'], sep="\t")
    train_df.to_csv(config.OUT_FILES['assertion_label_modified_train'], sep="\t")
    
    '''
    print('=================================')

    dev_df['label'] = dev_df['label'].apply(lambda ast_label: replace_ast_3_label(ast_label))
    test_df['label'] = test_df['label'].apply(lambda ast_label: replace_ast_3_label(ast_label))
    train_df['label'] = train_df['label'].apply(lambda ast_label: replace_ast_3_label(ast_label))

    print('Train data (modified label counts)')
    print(train_df['label'].value_counts())
    print('---------------------------------')
    print('Dev data (modified label counts)')
    print(dev_df['label'].value_counts())
    print('---------------------------------')
    print('Test data (modified label counts)')
    print(test_df['label'].value_counts())

    # dev_df.to_csv(config.OUT_FILES['assertion_3_label_modified_dev'], sep="\t")
    # test_df.to_csv(config.OUT_FILES['assertion_3_label_modified_test'], sep="\t")
    # train_df.to_csv(config.OUT_FILES['assertion_3_label_modified_train'], sep="\t")


if __name__ == "__main__":
    main()
