import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from ner_config import *
from data_processor import i2b2Dataset, pad_batch

train_data = pd.read_csv(data_path_train, sep="\t").astype(str)
dev_data = pd.read_csv(data_path_dev, sep="\t").astype(str)
test_data = pd.read_csv(data_path_test, sep="\t").astype(str)

train_dataset = i2b2Dataset(train_data)
train_iter = DataLoader(dataset=train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        collate_fn=pad_batch,
                        pin_memory=True
                        )
dev_dataset = i2b2Dataset(dev_data)
dev_iter = DataLoader(dataset=dev_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        collate_fn=pad_batch,
                        pin_memory=True
                        )
test_dataset = i2b2Dataset(test_data)
test_iter = DataLoader(dataset=test_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        collate_fn=pad_batch,
                        pin_memory=True
                        )