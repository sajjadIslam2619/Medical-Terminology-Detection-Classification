#####################################################################################################
# This model from hugging-face for assertion task. It is already fine-tuned with i2b2 2010 data. ####
# https://huggingface.co/bvanaken/clinical-assertion-negation-bert ##################################
#####################################################################################################

import os
import re
import numpy as np
import pandas as pd
from util import config
import time


# model from hugging_face :: bvanaken/clinical-assertion-negation-bert
def bvanaken_clinical_assertion_negation_bert(input):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("bvanaken/clinical-assertion-negation-bert")
    model = AutoModelForSequenceClassification.from_pretrained("bvanaken/clinical-assertion-negation-bert")

    tokenized_input = tokenizer(input, return_tensors="pt")
    output = model(**tokenized_input)

    predicted_label = np.argmax(output.logits.detach().numpy())

    if predicted_label == 0:
        return 'present'
    elif predicted_label == 1:
        return 'absent'
    elif predicted_label == 2:
        return 'possible'
    return ''


def main():
    return


if __name__ == "__main__":
    # main()

    input_str = "The patient recovered during the night and now suffers from [entity] shortness of breath [entity]."  # Present
    input_str = "She has a questionable [entity] sensitivity [entity] to iodine ."  # Possible -> Present
    input_str = 'Patient denies [entity] fever [entity].'  # Absent
    input_str = 'After transfer to the Floor , the patient was continued on Levofloxacin for a ten to 14 day course ' \
                'for presumptive [entity] pneumonia [entity] .'  # Possible -> Present
    input_str = 'Abdomen is soft and obese , [entity] nontender [entity] .'  # Absent
    input_str = 'Neurology concluded episode was most concerning for [entity] TIA [entity] .'  # Possible -> Present
    input_str = 'Slight irregularity of the origin of the right vertebral artery may be [entity] atherosclerotic [' \
                'entity] in nature .'  # Possible -> Possible

    # bvanaken_clinical_assertion_negation_bert(input_str)
    df = pd.read_csv(config.OUT_FILES['assertion_label_dev'], sep='\t', header=0)
    df = df.head(100)
    sentence_list = df['sentence'].tolist()
    label_list = df['label'].tolist()

    counter = 0
    for index, input_sentence in enumerate(sentence_list):
        print("index :: ", index)
        trained_label = label_list[index]
        # if index % 10 == 0: time.sleep(10.0)
        predicted_label = bvanaken_clinical_assertion_negation_bert(input_sentence)
        print(trained_label, predicted_label)
        if trained_label != predicted_label:
            counter += 1

    print("Wrong prediction :: ", counter)
