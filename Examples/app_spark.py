from flask import Flask, render_template, request, flash

import random
from werkzeug.utils import secure_filename

from processor import *

from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp

park = sparknlp.start(spark32=True)

from transformers import BertForTokenClassification, BertTokenizer

num_labels = len(tag2idx)
save_model_address = './trained_models/C-Bert-test'
model = BertForTokenClassification.from_pretrained(save_model_address, num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained(save_model_address, do_lower_case=False)
output_eval_file = "eval_results.txt"

UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = {'txt'}

documenter = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentencerDL = SentenceDetectorDLModel \
    .pretrained("sentence_detector_dl", "en") \
    .setInputCols(["document"]) \
    .setOutputCol("sentences")

sd_pipeline = PipelineModel(stages=[documenter, sentencerDL])
sd_model = LightPipeline(sd_pipeline)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_long_text(long_text):
    all_sentences = []
    all_tags = []
    for anno in sd_model.fullAnnotate(long_text)[0]["sentences"]:
        test_query = anno.result.replace('\n', '')
        print(test_query)
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
    return all_sentences, all_tags


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
        all_sentences, all_tags = predict_long_text(long_text)

    return render_template("pred_result.html", all_sentences=all_sentences, all_tags=all_tags)


@app.route('/predict_message', methods=['POST'])
def predict_message():
    if request.method != 'POST':
        return None
    message = request.form['message']
    all_sentences, all_tags = predict_long_text(message)
    return render_template("pred_result.html", all_sentences=all_sentences, all_tags=all_tags)


@app.route('/evaluate', methods=['POST'])
def evaluate():
    # randomly select a file 
    f_name = random.choice(os.listdir(config.INDIVIDUAL_TEST))
    f_path = os.path.join(config.INDIVIDUAL_TEST, f_name)
    dataframe = pd.read_csv(f_path, sep="\t").astype(str)
    sentences, labels = get_sentence_label(dataframe)
    input_ids, input_tags, attention_masks = process_data(sentences, labels, tokenizer)
    # temp_token, _, _ = create_query(sentences, tokenizer)
    query_inputs = torch.tensor(input_ids)
    query_tags = torch.tensor(input_tags)
    query_masks = torch.tensor(attention_masks)
    y_true, y_pred = model_evaluation(query_inputs, query_tags, query_masks, model)
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
