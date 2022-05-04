from flask import Flask, render_template, request, flash
import jinja2
env = jinja2.Environment()
env.globals.update(zip=zip)
import os
import random
from werkzeug.utils import secure_filename
import copy
from NER.processor import *
from NER.ner_utils import *
from NER import config
from Assertion.assertion_utils import *
from utils import *
from seqeval.metrics import classification_report, accuracy_score, f1_score
import stanza
try:
    nlp = stanza.Pipeline(lang='en', processors='tokenize')
except Exception:
    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize')

from transformers import BertForTokenClassification, BertTokenizer

model_ner, tokenizer_ner = load_ner_model()
model_assertion, tokenizer_assertion = load_assertion_model()
output_eval_file = "Results/eval_results.txt"

UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# predict entity


def predict_entities(long_text):
    all_sentences = []
    all_tags = []
    doc = nlp(long_text)
    for i, sentence in enumerate(doc.sentences):
        # temp_token: tokenized words
        # input_ids: convert temp_token to id
        temp_token, input_ids, attention_masks = create_query(
            sentence, tokenizer_ner)
        result_list = model_inference(model_ner, input_ids)
        result = [tag2name[t] for t in result_list]
        pretok_sent = ""
        pretags = ""
        for i, tok in enumerate(temp_token):
            if tok.startswith("##"):
                pretok_sent += tok[2:]
            else:
                pretok_sent += f" {tok}"
                pretags += f" {result[i]}"
        pretok_sent = pretok_sent[1:]
        pretags = pretags[1:]
        s = pretok_sent.split()
        t = pretags.split()
        all_sentences.append(s)
        all_tags.append(t)
    return all_sentences, all_tags

# extract index of entity
# extract sentence with assertion


def predict_assertion(all_sentences, all_tags):
    sentences_with_problem, \
        all_problems_in_text, \
        all_treatment_in_text, \
        all_test_in_text = entity_extractor(all_sentences, all_tags)
    input_ids, attention_mask = assertion_input_creator(
        sentences_with_problem, tokenizer_assertion, add_special_tokens=False)
    pred_labels = assertion_model_inference(
        model_assertion, input_ids, attention_mask)
    # map labels wordwisely.
    """
    e.g. problem_list = [[[4, 5, 6], [10]], '', [[2, 3], [6,7]]]
        label = ['yes', 'no', 'what', 'yes']
    resutls: [['yes', 'yes', 'yes', 'no'], '', ['what', 'what', 'yes', 'yes']]
    """
    i_label = 0
    labels_in_sentence = []
    for index in all_problems_in_text:
        if index:
            tmp = []
            for i_p in index:
                tmp.extend([pred_labels[i_label]] * len(i_p))
                i_label += 1
            labels_in_sentence.append(tmp)
        else:
            labels_in_sentence.append(index)
            
    all_problems_in_text_flatten = list(map(lambda l:
                                            [int(item) for elem in l for item in elem], all_problems_in_text))
    return labels_in_sentence, all_problems_in_text_flatten, all_treatment_in_text, all_test_in_text


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_file', methods=['POST'])
def predict_file():
    all_sentences, all_tags = [], []
    if request.method != 'POST':
        return None
        # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return None
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return None
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file.seek(0)
        long_text = file.read()
        long_text = str(long_text, 'utf-8')
        all_sentences, all_tags = predict_entities(long_text)
        
        assertion_in_sentence, \
        all_problems_in_text_flatten, \
        all_treatment_in_text, \
        all_test_in_text = predict_assertion(all_sentences, all_tags)
        
    return render_template("pred_result.html", all_sentences=all_sentences,
                           labels_in_sentence=assertion_in_sentence,
                           all_problems_in_text=all_problems_in_text_flatten,
                           all_treatment_in_text=all_treatment_in_text,
                           all_test_in_text=all_test_in_text,
                           )


@app.route('/predict_message', methods=['POST'])
def predict_message():
    if request.method != 'POST':
        return None
    message = request.form['message']
    all_sentences, all_tags = predict_entities(message)
    assertion_in_sentence, \
    all_problems_in_text_flatten, \
    all_treatment_in_text, \
    all_test_in_text = predict_assertion(all_sentences, all_tags)
        
    # for sentence, pred_assertion, assertion_index,\
    # treatment_index, test_index in zip(all_sentences,
    #                                    assertion_in_sentence, all_problems_in_text_flatten,
    #                                    all_treatment_in_text,
    #                                    all_test_in_text):
    #     for i, word in enumerate(sentence):
    #         if i in assertion_index:
    #             index = assertion_index.index(i)
    #             if pred_assertion[index] == 'Present':
    #                 print(f'||| {word} |||', end=' ')
    #             elif pred_assertion[index] == 'Possible':
    #                 print(f'--- {word} ---', end=' ')
    #             elif pred_assertion[index] == 'Not-present':
    #                 print(f'+++ {word} +++', end=' ')
    #         elif i in treatment_index:
    #             print(f'### {word} ###', end=' ')
    #         elif i in test_index:
    #             print(f'@@@ {word} ###', end=' ')
    #         else:
    #             print(word, end=' ')   
        
    return render_template("pred_result.html", all_sentences=all_sentences,
                            assertion_in_sentence=assertion_in_sentence,
                            all_problems_in_text=all_problems_in_text_flatten,
                            all_treatment_in_text=all_treatment_in_text,
                            all_test_in_text=all_test_in_text,
                            )


@app.route('/evaluate', methods=['POST'])
def evaluate():
    # randomly select a file
    f_name = random.choice(os.listdir(config.INDIVIDUAL_TEST))
    f_path = os.path.join(config.INDIVIDUAL_TEST, f_name)
    dataframe = pd.read_csv(f_path, sep="\t").astype(str)
    sentences, labels = get_sentence_label(dataframe)
    input_ids, input_tags, attention_masks = process_data(
        sentences, labels, tokenizer_ner)
    # temp_token, _, _ = create_query(sentences, tokenizer)
    query_inputs = torch.tensor(input_ids)
    query_tags = torch.tensor(input_tags)
    query_masks = torch.tensor(attention_masks)
    y_true, y_pred = model_evaluation(
        query_inputs, query_tags, query_masks, model_ner)
    # Get acc , recall, F1 result report
    report = classification_report(y_true, y_pred, digits=4)
    # Save the report into file
    acc = '{:.2f}'.format(accuracy_score(y_true, y_pred))
    f1 = '{:.2f}'.format(f1_score(y_true, y_pred))

    with open(output_eval_file, "w") as writer:
        print("***** Eval results *****")
        print("\n%s" % (report))
        print("f1 socre: ", f1)
        print("Accuracy score: ", acc)

        writer.write("f1 socre:\n")
        writer.write(f1)
        writer.write("\n\nAccuracy score:\n")
        writer.write(acc)
        writer.write("\n\n")
        writer.write(report)

    print(f_name)

    df = classification_report_to_dataframe(report)

    return render_template("eval_result.html",
                           file_name=os.path.basename(f_name),
                           sentences=sentences, true_label=y_true,
                           pred_label=y_pred,
                           acc=acc, f1=f1,
                           column_names=df.columns.values,
                           row_data=list(df.values.tolist()),
                           zip=zip
                           )

    # return render_template("eval_result.html", sentences=sentences,
    #                        true_label=y_true, pred_label=y_pred,
    #                        file_name=os.path.basename(f_name))


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8880)
