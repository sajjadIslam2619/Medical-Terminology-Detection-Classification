import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)


def assertion_input_creator(sentences, tokenizer, add_special_tokens=True):
    input_ids = []
    attention_mask = []
    # For every sentence...
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,                      # Sentence to encode.
            add_special_tokens=add_special_tokens,  # Add '[CLS]' and '[SEP]'
            max_length=128,           # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,   # Construct attn. masks.
            return_tensors='pt',     # Return pytorch tensors.
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_mask.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    return torch.tensor(input_ids), torch.tensor(attention_mask)


def assertion_model_inference(model, input_ids, attention_mask):
    model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    predictions, true_labels = [], []
    model.eval()
    with torch.no_grad():
        result = model(input_ids, token_type_ids=None,
                       attention_mask=attention_mask, return_dict=True)

    logits = result.logits
    logits = logits.detach().cpu().numpy()
    predictions.append(logits)

    pred_labels_i = np.argmax(logits, axis=1).flatten()

    def index2label(x):
        if x == 0:
            return 'Present'
        elif x == 1:
            return 'Possible'
        elif x == 2:
            return 'Conditional'
        elif x == 3:
            return 'Associated with someone else'
        elif x == 4:
            return 'Hypothetical'
        elif x == 5:
            return 'Absent'

    pred_labels = map(index2label, pred_labels_i)
    return list(pred_labels)
