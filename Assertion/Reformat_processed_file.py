import os, re, pickle
import numpy as np
import pandas as pd
import json
from util import config

np.random.seed(1)


def replaceAssertionTag(ast_tag):
    if ast_tag == 'I-present':
        return 'I-present'
    elif ast_tag == 'B-present':
        return 'B-present'
    elif ast_tag == 'O':
        return 'O'
    elif ast_tag.startswith('B-'):
        return 'B-absent'
    elif ast_tag.startswith('I-'):
        return 'I-absent'
    return ''


def main():
    dev_df = pd.read_csv(config.OUT_FILES['merged_dev'], sep='\t', header=0)
    test_df = pd.read_csv(config.OUT_FILES['merged_test'], sep='\t', header=0)
    train_df = pd.read_csv(config.OUT_FILES['merged_train'], sep='\t', header=0)

    dev_df['ast'] = dev_df['ast'].apply(lambda ast_tag: replaceAssertionTag(ast_tag))
    test_df['ast'] = test_df['ast'].apply(lambda ast_tag: replaceAssertionTag(ast_tag))
    train_df['ast'] = train_df['ast'].apply(lambda ast_tag: replaceAssertionTag(ast_tag))

    dev_df.to_csv(config.OUT_FILES['merged_ast_dev'], sep="\t")
    test_df.to_csv(config.OUT_FILES['merged_ast_test'], sep="\t")
    train_df.to_csv(config.OUT_FILES['merged_ast_train'], sep="\t")


if __name__ == "__main__":
    main()
