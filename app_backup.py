from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
from scipy.special import softmax

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pyspark.ml import PipelineModel
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp

import config
import utils
import processor

tag2idx=processor.tag2idx

max_len  = 128
model_name = 'C-Bert-autobert'

app = Flask(__name__)
spark = sparknlp.start(spark32 = True)

documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
sentencerDL = SentenceDetectorDLModel\
    .pretrained("sentence_detector_dl", "en") \
    .setInputCols(["document"]) \
    .setOutputCol("sentences")

sd_pipeline = PipelineModel(stages=[documenter, sentencerDL])
sd_model = LightPipeline(sd_pipeline)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method != 'POST':
		return None
	message = request.form['message']
	all_sentences = []
	all_tags = []
	for anno in sd_model.fullAnnotate(message)[0]["sentences"]:
		test_query = anno.result.replace('\n','')

		tokenized_texts = []
		temp_token = ['[CLS]']
		token_list = tokenizer.tokenize(test_query)
		temp_token.extend(token_list)
		temp_token = temp_token[:max_len-1]
		temp_token.append('[SEP]')
		input_id = tokenizer.convert_tokens_to_ids(temp_token)
		padding_len = max_len - len(input_id)
		input_id = input_id + ([0] * padding_len)
		tokenized_texts = [input_id]
		attention_masks = [[int(i>0) for i in input_id]]
		tokenized_texts = torch.tensor(tokenized_texts)
		attention_masks = torch.tensor(attention_masks)
		# Set save model to Evalue loop
		model.eval()
		# Get model predict result
		with torch.no_grad():
			outputs = model(tokenized_texts, token_type_ids=None,
			attention_mask=None,)
			# For eval mode, the first result of outputs is logits
			logits = outputs[0]
		predict_results = logits.detach().cpu().numpy()
		result_arrays_soft = softmax(predict_results[0])
		result_list = np.argmax(result_arrays_soft, axis=-1)
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

	return render_template("result.html", all_sentences=all_sentences, all_tags=all_tags)


if __name__ == '__main__':
	# app.run(debug=True)
	app.run(host='0.0.0.0', port=8880)