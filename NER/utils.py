import numpy as np

from io import BytesIO
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.transforms import IdentityTransform

from scipy.special import softmax

import torch.nn.functional as F

from NER.processor import *


def model_evaluation(input_ids, label_ids, input_mask, model):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None,
        attention_mask=input_mask)
        # For eval mode, the first result of outputs is logits
        logits = outputs[0]
    # Get NER predict result
    logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
    logits = logits.detach().cpu().numpy()
    # Get NER true result
    label_ids = label_ids.to('cpu').numpy()
    # Only predict the real word, mark=0, will not calculate
    input_mask = input_mask.to('cpu').numpy()
    # Compare the valuable predict result
    for i, mask in enumerate(input_mask):
        # Real one
        temp_1 = [] 
        # Predict one
        temp_2 = []
        for j, m in enumerate(mask):
            # Mark=0, meaning its a pad word, dont compare
            if m:
                if tag2name[label_ids[i][j]] not in ["X", "[CLS]", "[SEP]"]: # Exclude the X label
                    # print(tag2name[logits[i][j]])
                    temp_1.append(tag2name[label_ids[i][j]])
                    temp_2.append(tag2name[logits[i][j]])
            else:
                break
        y_true.append(temp_1)
        y_pred.append(temp_2)

    return y_true, y_pred


def model_inference(model, input_ids):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None,
        attention_mask=None)
        # For eval mode, the first result of outputs is logits
        logits = outputs[0]
    # Get NER predict result
    predict_results = logits.detach().cpu().numpy()
    result_arrays_soft = softmax(predict_results[0])
    
    return np.argmax(result_arrays_soft, axis=-1)

def txt_to_pic(txt_path):
    def text_to_rgba(s, *, dpi, **kwargs):
        # To convert a text string to an image, we can:
        # - draw it on an empty and transparent figure;
        # - save the figure to a temporary buffer using ``bbox_inches="tight",
        #   pad_inches=0`` which will pick the correct area to save;
        # - load the buffer using ``plt.imread``.
        #
        # (If desired, one can also directly save the image to the filesystem.)
        fig = Figure(facecolor="none")
        fig.text(0, 0, s, **kwargs)
        with BytesIO() as buf:
            fig.savefig(buf, dpi=dpi, format="png", bbox_inches="tight",
                        pad_inches=None)
            buf.seek(0)
            rgba = plt.imread(buf)
        return rgba


    fig = plt.figure()
    rgba1 = text_to_rgba(r"Metrics", color="black", fontsize=10, dpi=100)
    # rgba2 = text_to_rgba(r"some other string", color="red", fontsize=20, dpi=200)
    # One can then draw such text images to a Figure using `.Figure.figimage`.
    fig.figimage(rgba1, 30, 440)
    # fig.figimage(rgba2, 100, 150)

    # One can also directly draw texts to a figure with positioning
    # in pixel coordinates by using `.Figure.text` together with
    # `.transforms.IdentityTransform`.
    # fig.text(100, 250, r"IQ: $\sigma_i=15$", color="black", fontsize=20,
    #          transform=IdentityTransform())
    # fig.text(100, 350, r"some other string", color="red", fontsize=20,
    #          transform=IdentityTransform())

    with open(txt_path) as f:
        content = f.readlines()

    for i, c in enumerate(content):
        c = c.strip('\n')
        fig.text(50, 400-i*25, c, color="black", fontsize=10,
                transform=IdentityTransform())
        
    plt.savefig("static/results.png")
    
    
def classification_report_to_dataframe(str_representation_of_report):
    split_string = [x.split(' ') for x in str_representation_of_report.split('\n')]
    column_names = [' ']+[x for x in split_string[0] if x!='']
    values = []
    for table_row in split_string[1:-1]:
        table_row = [value for value in table_row if value!='']
        if table_row!=[]:
            values.append(table_row)
    for i in values:
        for j in range(len(i)):
            if i[1] == 'avg':
                i[0:2] = [''.join(i[0:2])]
            if len(i) == 3:
                i.insert(1, np.nan)
                i.insert(2, np.nan)
            else:
                pass
    return pd.DataFrame(data=values, columns=column_names)