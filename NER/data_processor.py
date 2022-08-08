import sys
# sys.path.append("..")
import torch
from torch.utils.data import Dataset
from ner_config import tag2idx, BASE_MODEL, MAX_LEN
from transformers import BertTokenizer, AutoTokenizer


class i2b2Dataset(Dataset):
    def __init__(self, dataframe):
        self.sentences = []
        self.labels = []
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.MAX_LEN = MAX_LEN - 2
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),
                                                     s["tag"].values.tolist())]
        grouped = dataframe.groupby("sentence #").apply(agg_func)
        sentences_labels = [s for s in grouped]

        self.sentences = [[s[0] for s in sent] for sent in sentences_labels]
        self.labels = [[s[1] for s in sent] for sent in sentences_labels]

    def __getitem__(self, idx):
        sentence, label = self.sentences[idx], self.labels[idx]
        temp_lable = []
        temp_token = []
        for word, lab in zip(sentence, label):
            token_list = self.tokenizer.tokenize(word)
            for m, token in enumerate(token_list):
                temp_token.append(token)
                if m == 0:
                    temp_lable.append(lab)
                else:
                    temp_lable.append('X')
        text = ['[CLS]'] + temp_token[:self.MAX_LEN] + ['[SEP]']
        label = ['[CLS]'] + temp_lable[:self.MAX_LEN] + ['[SEP]']
        # convert to ids
        sentence_ids = self.tokenizer.convert_tokens_to_ids(text)
        label_ids = [tag2idx.get(t) for t in label]
        seqlen = len(label_ids)
        return sentence_ids, label_ids, seqlen

    def __len__(self):
        return len(self.sentences)


def pad_batch(batch):
    maxlen = max([i[2] for i in batch])
    token_tensors = torch.LongTensor([i[0] + [0] * (maxlen - len(i[0])) for i in batch])
    #  Label of <PAD>: 0
    label_tensors = torch.LongTensor([i[1] + [tag2idx.get('O')] * (maxlen - len(i[1])) for i in batch])
    mask = (token_tensors > 0)
    return token_tensors, label_tensors, mask