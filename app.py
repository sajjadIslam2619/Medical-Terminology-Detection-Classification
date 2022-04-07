from lib2to3.pgen2 import token
import re
from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
from scipy.special import softmax
import os  
import random
import torch


import config
from utils import *
from processor import *

from pyspark.ml import PipelineModel
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
park = sparknlp.start(spark32 = True)

from transformers import BertForTokenClassification, BertTokenizer
num_labels = len(tag2idx)
save_model_address = './trained_models/C-Bert-test'
model = BertForTokenClassification.from_pretrained(save_model_address, num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained(save_model_address, do_lower_case=False)
output_eval_file = "eval_results.txt"

documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
sentencerDL = SentenceDetectorDLModel\
    .pretrained("sentence_detector_dl", "en") \
    .setInputCols(["document"]) \
    .setOutputCol("sentences")

sd_pipeline = PipelineModel(stages=[documenter, sentencerDL])
sd_model = LightPipeline(sd_pipeline)

    
app = Flask(__name__)



@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method != 'POST':
        return None
    all_sentences = []
    all_tags = []
    message = request.form['message']
    all_sentences = []
    all_tags = []
    for anno in sd_model.fullAnnotate(message)[0]["sentences"]:
        test_query = anno.result.replace('\n','')
        # temp_token: tokenized words
        # input_ids: convert temp_token to id
        temp_token, input_ids, attention_masks = create_query(test_query, tokenizer)
        result_list = model_inference(model, input_ids)
        result = [tag2name[t] for t in result_list]
        pretok_sent = ""
        pretags = ""
        for i, tok in enumerate(temp_token):
            if tok.startswith("##"):
                pretok_sent += tok[2:]
            else:
                pretok_sent += " " + tok
                pretags += " " + result[i]
        pretok_sent = pretok_sent[1:]
        pretags = pretags[1:]

        s = pretok_sent.split()
        t = pretags.split()

        all_sentences.append(s)
        all_tags.append(t)

    return render_template("pred_result.html", all_sentences=all_sentences, all_tags=all_tags)
        
    
@app.route('/evaluate', methods=['POST'])
def evaluate():
    # randomly select a file 
    f_name = random.choice(os.listdir(config.INDIVIDUAL_TEST))
    f_path = os.path.join(config.INDIVIDUAL_TEST, f_name) 
    dataframe = pd.read_csv(f_path, sep="\t").astype(str)
    sentences, labels = get_sentence_label(dataframe)
    input_ids, input_tags, attention_masks = process_data(sentences, labels, tokenizer)
    print(sentences)
    query_inputs = torch.tensor(input_ids)
    query_tags = torch.tensor(input_tags)
    query_masks = torch.tensor(attention_masks)
    y_true, y_pred = model_evaluation(query_inputs, query_tags, query_masks, model)
    # Get acc , recall, F1 result report
    report = classification_report(y_true, y_pred, digits=4)
    # Save the report into file

    with open(output_eval_file, "w") as writer:
        print("***** Eval results *****")
        print("\n%s"%(report))
        print("f1 socre: %f"%(f1_score(y_true, y_pred)))
        print("Accuracy score: %f"%(accuracy_score(y_true, y_pred)))

        writer.write("f1 socre:\n")
        writer.write(str(f1_score(y_true, y_pred)))        
        writer.write("\n\nAccuracy score:\n")
        writer.write(str(accuracy_score(y_true, y_pred)))
        writer.write("\n\n")  
        writer.write(report)
    print(f_name)
    txt_to_pic(output_eval_file)
    
    
    return render_template("eval_result.html", sentences=sentences, true_label=y_true, pred_label=y_pred, file_name=os.path.basename(f_name))

if __name__ == '__main__':
	# app.run(debug=True)
	app.run(host='0.0.0.0', port=8880)