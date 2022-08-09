import pandas as pd
import torch
#import NER.config as config
import config

tag2idx={'B-problem': 0,
        'B-test': 1,
        'B-treatment': 2,
        'I-problem': 3,
        'I-test': 4,
        'I-treatment': 5,
        'O': 6,
        'X': 7,
        '[CLS]': 8,
        '[SEP]': 9
}
tag2name = {tag2idx[key]: key for key in tag2idx}

def get_sentence_label(dataframe):
    
    agg_func = lambda s: list(zip(s["word"].values.tolist(), s["tag"].values.tolist()))
    grouped = dataframe.groupby("sentence #").apply(agg_func)
    sentences_labels = list(grouped)
    # Get sentence data
    sentences = [[s[0] for s in sent] for sent in sentences_labels]
    labels = [[l[1] for l in label] for label in sentences_labels]
    return sentences, labels


def process_data(sentences, labels, tokenizer):
    tokenized_texts = []
    word_piece_labels = []
    for word_list,label in (zip(sentences, labels)):
        temp_lable = []
        temp_token = []
        # tokenize in wordpiece 
        # add labels to tokens
        for word,lab in zip(word_list,label):
            token_list = tokenizer.tokenize(word)
            for m,token in enumerate(token_list):
                temp_token.append(token)
                if m==0:
                    temp_lable.append(lab)
                else:
                    temp_lable.append('X')  

        tokenized_texts.append(temp_token)
        word_piece_labels.append(temp_lable)
    id_list = []
    target_list = []
    attention_mask_list = []
    # PADING
    for text, label in zip(tokenized_texts, word_piece_labels):
        
        # Add [CLS] and [SEP], 
        # Truncate seq if it is too long
        text = ['[CLS]'] + text[:config.MAX_LEN - 2] + ['[SEP]']
        label = ['[CLS]'] + label[:config.MAX_LEN - 2] + ['[SEP]']

        # convert to ids
        ids = tokenizer.convert_tokens_to_ids(text)
        target_tag =[tag2idx.get(t) for t in label]

        # padding 
        # Label [PAD] with O (other)
        padding_len = config.MAX_LEN - len(ids)
        ids = ids + [0] * padding_len
        target_tag += [tag2idx['O']] * padding_len

        # create masks
        attention_masks = [int(i>0) for i in ids]

        id_list.append(ids)
        target_list.append(target_tag)
        attention_mask_list.append(attention_masks)

    return id_list, target_list, attention_mask_list


# input a whole sentence as a string
# like 'He was admitted , taken to the operating room where he underwent L5-S1 right hemilaminectomy and discectomy .'
def create_query(sentence, tokenizer):

    temp_token = ['[CLS]']
    # word_list = [token.text for token in sentence.tokens]
    for word in sentence:
        temp_token.extend(tokenizer.tokenize(word))
    temp_token = temp_token[:128 - 1]
    temp_token.append('[SEP]')
    input_id = tokenizer.convert_tokens_to_ids(temp_token)
    padding_len = config.MAX_LEN - len(input_id)
    input_id = input_id + ([0] * padding_len)
    tokenized_texts = [input_id]
    attention_masks = [[int(i>0) for i in input_id]]

    return temp_token, torch.tensor(tokenized_texts), torch.tensor(attention_masks)


def main():
    
    data_path_train = "../Data/processed/merged/train.tsv"
    train_data = pd.read_csv(data_path_train, sep="\t").astype(str)
    s, l = get_sentence_label(train_data)
    
    print(s[104])
    print(l[104])
    
if __name__ == "__main__":
    main()