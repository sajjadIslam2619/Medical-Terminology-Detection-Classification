import torch
import numpy as np
from sklearn import metrics
from ner_config import *


def train(epoch, model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = 0.0
    step = 0
    for i, batch in enumerate(data_loader):
        optimizer.zero_grad()
        step += 1
        contexts, labels, masks = batch
        contexts = contexts.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        loss = model(contexts, labels, masks)
        losses += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    print("Epoch: {}, Loss:{:.4f}".format(epoch, losses / step))

def validate(epoch, model, data_loader, device):
    model.eval()
    Y, Y_hat = [], []
    losses = 0
    step = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            step += 1
            contexts, labels, masks = batch
            contexts = contexts.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            y_hat = model(contexts, labels, masks, is_test=True)
            loss = model(contexts, labels, masks)

            losses += loss.item()
            # Save prediction
            for j in y_hat:
                # 1-dimension
                Y_hat.extend(j)
            # Save labels
            masks = (masks == 1)
            y_orig = torch.masked_select(labels, masks)
            Y.append(y_orig.cpu())
    # 2-dimension --> 1-dimension
    Y = torch.cat(Y, dim=0).numpy()
    Y_hat = np.array(Y_hat)
    acc = (Y_hat == Y).mean() * 100

    print("Epoch: {}, Val Loss:{:.4f}, Val Acc:{:.3f}%".format(epoch, losses / step, acc))
    return model, losses / step, acc


def test(model, data, device):
    model.eval()
    Y, Y_hat = [], []
    with torch.no_grad():
        for i, batch in enumerate(data):
            contexts, labels, masks = batch
            contexts = contexts.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            y_hat = model(contexts, labels, masks, is_test=True)
            # Save prediction
            for j in y_hat:
                Y_hat.extend(j)
            # Save labels
            masks = (masks == 1)
            y_orig = torch.masked_select(labels, masks)
            Y.append(y_orig.cpu())

    Y = torch.cat(Y, dim=0).numpy()
    y_true = [idx2tag[i] for i in Y]
    y_pred = [idx2tag[i] for i in Y_hat]
    
    print(metrics.classification_report(y_true, y_pred, labels=LABELS, digits=4))
    return y_true, y_pred

def infer(model, tokenizer, sentence):
    sentence_ids = tokenizer.convert_tokens_to_ids(sentence)
    sentence_tensor = torch.LongTensor(sentence_ids).unsqueeze(0).to(DEVICE)
    mask = (sentence_tensor > 0)
    y = model(sentence_tensor, None, mask, is_test=True)
    y = y[0]
    y_tag = [idx2tag[i] for i in y]
    y_tag_cn = []
    for i in range(len(y_tag)):
        if y_tag[i] in ['<PAD>', '[CLS]', '[SEP]', 'O']:
            y_tag_cn.append('O')
        else:
            y_tag_cn.append(idx2tag[y_tag[i][2:]])
            
    y_tag_cn.append('O')
    result = []
    i = 0
    cur_tag = 'None'
    while i < len(y_tag_cn):
        if y_tag_cn[i] != 'O':
            pos_start = i
            cur_tag = y_tag_cn[i]
            pos_end = i + 1
            while pos_end < len(y_tag_cn):
                if y_tag_cn[pos_end] == cur_tag:
                    pos_end += 1
                else:
                    break
            result.append([cur_tag, sentence[pos_start:pos_end], pos_start, pos_end])
            i = pos_end
        else:
            i += 1
    print("The result are shown below:")
    for entity in result:
        entity_name = ''.join(entity[1])
        print(f"entity_name:{entity_name}, entity_type:{entity[0]}, start_pos:{entity[2]}, end_pos:{entity[3]}")
