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
    nlp = stanza.Pipeline(lang="en", processors="tokenize")
except Exception:
    stanza.download("en")
    nlp = stanza.Pipeline(lang="en", processors="tokenize")

model_ner, tokenizer_ner = load_ner_model()
model_assertion, tokenizer_assertion = load_assertion_model()
output_eval_file = "./Results/eval_results.txt"

UPLOAD_FOLDER = "./Results"
ALLOWED_EXTENSIONS = {"txt"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_entities(long_text):
    all_sentences = []
    all_tags = []
    doc = nlp(long_text)
    for i, sentence in enumerate(doc.sentences):
        # temp_token: tokenized words
        # input_ids: convert temp_token to id
        word_list = [token.text for token in sentence.tokens]
        temp_token, input_ids, attention_masks = create_query(word_list, tokenizer_ner)
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


def predict_assertion(all_sentences, all_tags):

    # extract index of entity
    # extract sentence with assertion

    (
        sentences_with_problem,
        all_problems_in_text,
        all_treatment_in_text,
        all_test_in_text,
    ) = entity_extractor(all_sentences, all_tags)
    input_ids, attention_mask = assertion_input_creator(
        sentences_with_problem, tokenizer_assertion, add_special_tokens=False
    )
    pred_labels = assertion_model_inference(model_assertion, input_ids, attention_mask)
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

    all_problems_in_text_flatten = list(
        map(lambda l: [int(item) for elem in l for item in elem], all_problems_in_text)
    )
    return (
        labels_in_sentence,
        all_problems_in_text_flatten,
        all_treatment_in_text,
        all_test_in_text,
    )


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_file", methods=["POST"])
def predict_file():
    all_sentences, all_tags = [], []
    if request.method != "POST":
        return None
        # check if the post request has the file part
    if "file" not in request.files:
        flash("No file part")
        return None
    file = request.files["file"]
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == "":
        flash("No selected file")
        return None
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        file.seek(0)
        long_text = file.read()
        long_text = str(long_text, "utf-8")
        all_sentences, all_tags = predict_entities(long_text)

        (
            assertion_in_sentence,
            all_problems_in_text_flatten,
            all_treatment_in_text,
            all_test_in_text,
        ) = predict_assertion(all_sentences, all_tags)

    # These lists are to collect problem-entity label-wise and display on the UI Table
    list_ast_present_entity = []
    list_ast_absent_entity = []
    list_ast_posssible_entity = []
    list_ast_conditional_entity = []
    list_ast_hyphothetical_entity = []
    list_ast_associated_entity = []

    # This string is to format full clinical text and add <tag>, like  <Problem-present> problem-entity </Problem-present> 
    output_text_with_classification = ''

    for (
        sentence,
        tags,
        pred_assertion,
        assertion_index,
        treatment_index,
        test_index,
    ) in zip(
        all_sentences,
        all_tags,
        assertion_in_sentence,
        all_problems_in_text_flatten,
        all_treatment_in_text,
        all_test_in_text,
    ):

        for i, word in enumerate(sentence):
            
            if i in assertion_index:
                index = assertion_index.index(i)
                if (tags[i] == 'B-problem') : 
                    if pred_assertion[index] == "Present":
                        list_ast_present_entity.append(word)
                        output_text_with_classification = output_text_with_classification + " <Problem-present> " + word
                        # For one word problem-entity, there is only 'beginning' tag no 'inside' tag
                        # tags : 'O', 'O', 'B-problem', 'O', 'O',
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-present> " 
                    elif pred_assertion[index] == "Possible":
                        list_ast_posssible_entity.append(word)
                        output_text_with_classification = output_text_with_classification + " <Problem-possible> " + word
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-possible> " 
                    elif pred_assertion[index] == "Conditional":
                        list_ast_conditional_entity.append(word)
                        output_text_with_classification = output_text_with_classification + " <Problem-conditional> " + word
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-conditional> " 
                    elif pred_assertion[index] == "Hypothetical":
                        list_ast_hyphothetical_entity.append(word)
                        output_text_with_classification = output_text_with_classification + " <Problem-hypothetical> " + word
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-hypothetical> " 
                    elif pred_assertion[index] == "Associated with someone else":
                        list_ast_associated_entity.append(word)
                        output_text_with_classification = output_text_with_classification + " <Problem-associated> " + word
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-associated> " 
                    elif pred_assertion[index] == "Absent":
                        list_ast_absent_entity.append(word)
                        output_text_with_classification = output_text_with_classification + " <Problem-absent> " + word
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-absent> " 
                elif (tags[i] == 'I-problem'): 
                    if pred_assertion[index] == "Present":
                        list_ast_present_entity[-1] = list_ast_present_entity[-1] +" "+ word
                        output_text_with_classification = output_text_with_classification + " " + word 
                        # tags: 'O', 'B-problem', 'I-problem', 'I-problem', 'O',
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-present> " 
                    elif pred_assertion[index] == "Possible":
                        list_ast_posssible_entity[-1] = list_ast_posssible_entity[-1] +" "+ word
                        output_text_with_classification = output_text_with_classification + " " + word 
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-possible> " 
                    elif pred_assertion[index] == "Conditional":
                        list_ast_conditional_entity[-1] = list_ast_conditional_entity[-1] +" "+ word
                        output_text_with_classification = output_text_with_classification + " " + word 
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-conditional> " 
                    elif pred_assertion[index] == "Hypothetical":
                        list_ast_hyphothetical_entity[-1] = list_ast_hyphothetical_entity[-1] +" "+ word
                        output_text_with_classification = output_text_with_classification + " " + word 
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-hypothetical> " 
                    elif pred_assertion[index] == "Associated with someone else":
                        list_ast_associated_entity[-1] = list_ast_associated_entity[-1] +" "+ word
                        output_text_with_classification = output_text_with_classification + " " + word 
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-associated> " 
                    elif pred_assertion[index] == "Absent":
                        list_ast_absent_entity[-1] = list_ast_absent_entity[-1] +" "+ word
                        output_text_with_classification = output_text_with_classification + " " + word 
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-absent> " 
            elif i in treatment_index:
                if (tags[i] == 'B-treatment') : 
                    output_text_with_classification = output_text_with_classification + " <Treatment> " + word
                    # For one word Treatment-entity, there is only 'beginning' tag no 'inside' tag
                    if tags[i+1] != 'I-treatment':
                        output_text_with_classification = output_text_with_classification + " </Treatment> " 
                elif (tags[i] == 'I-treatment') :
                    output_text_with_classification = output_text_with_classification + " " + word 
                    if tags[i+1] != 'I-treatment':
                        output_text_with_classification = output_text_with_classification + " </Treatment> " 
            elif i in test_index:
                if (tags[i] == 'B-test') : 
                    output_text_with_classification = output_text_with_classification + " <Test> " + word
                    # For one word Test-entity, there is only 'beginning' tag no 'inside' tag
                    if tags[i+1] != 'I-test':
                        output_text_with_classification = output_text_with_classification + " </Test> " 
                elif (tags[i] == 'I-test') :
                    output_text_with_classification = output_text_with_classification + " " + word 
                    if tags[i+1] != 'I-test':
                        output_text_with_classification = output_text_with_classification + " </Test> " 
            else:
                if (word.strip() != '[SEP]' and word.strip() != '[CLS]'):
                    output_text_with_classification = output_text_with_classification + " " + word
    

    return render_template(
        "pred_result.html",
        all_sentences=all_sentences,
        assertion_in_sentence=assertion_in_sentence,
        all_problems_in_text=all_problems_in_text_flatten,
        all_treatment_in_text=all_treatment_in_text,
        all_test_in_text=all_test_in_text,
        list_ast_present_entity=list_ast_present_entity,
        list_ast_posssible_entity=list_ast_posssible_entity,
        list_ast_conditional_entity=list_ast_conditional_entity,
        list_ast_hyphothetical_entity=list_ast_hyphothetical_entity,
        list_ast_associated_entity=list_ast_associated_entity,
        list_ast_absent_entity=list_ast_absent_entity,
    )


@app.route("/predict_message", methods=["POST"])
def predict_message():
    if request.method != "POST":
        return None
    message = request.form["message"]
    all_sentences, all_tags = predict_entities(message)
    (
        assertion_in_sentence,
        all_problems_in_text_flatten,
        all_treatment_in_text,
        all_test_in_text,
    ) = predict_assertion(all_sentences, all_tags)

    # These lists are to collect problem-entity label-wise and display on the UI Table
    list_ast_present_entity = []
    list_ast_absent_entity = []
    list_ast_posssible_entity = []
    list_ast_conditional_entity = []
    list_ast_hyphothetical_entity = []
    list_ast_associated_entity = []

    # This string is to format full clinical text and add <tag>, like  <Problem-present> problem-entity </Problem-present> 
    output_text_with_classification = ''

    for (
        sentence,
        tags,
        pred_assertion,
        assertion_index,
        treatment_index,
        test_index,
    ) in zip(
        all_sentences,
        all_tags,
        assertion_in_sentence,
        all_problems_in_text_flatten,
        all_treatment_in_text,
        all_test_in_text,
    ):

        for i, word in enumerate(sentence):
            
            if i in assertion_index:
                index = assertion_index.index(i)
                if (tags[i] == 'B-problem') : 
                    if pred_assertion[index] == "Present":
                        list_ast_present_entity.append(word)
                        output_text_with_classification = output_text_with_classification + " <Problem-present> " + word
                        # For one word problem-entity, there is only 'beginning' tag no 'inside' tag
                        # tags : 'O', 'O', 'B-problem', 'O', 'O',
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-present> " 
                    elif pred_assertion[index] == "Possible":
                        list_ast_posssible_entity.append(word)
                        output_text_with_classification = output_text_with_classification + " <Problem-possible> " + word
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-possible> " 
                    elif pred_assertion[index] == "Conditional":
                        list_ast_conditional_entity.append(word)
                        output_text_with_classification = output_text_with_classification + " <Problem-conditional> " + word
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-conditional> " 
                    elif pred_assertion[index] == "Hypothetical":
                        list_ast_hyphothetical_entity.append(word)
                        output_text_with_classification = output_text_with_classification + " <Problem-hypothetical> " + word
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-hypothetical> " 
                    elif pred_assertion[index] == "Associated with someone else":
                        list_ast_associated_entity.append(word)
                        output_text_with_classification = output_text_with_classification + " <Problem-associated> " + word
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-associated> " 
                    elif pred_assertion[index] == "Absent":
                        list_ast_absent_entity.append(word)
                        output_text_with_classification = output_text_with_classification + " <Problem-absent> " + word
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-absent> " 
                elif (tags[i] == 'I-problem'): 
                    if pred_assertion[index] == "Present":
                        list_ast_present_entity[-1] = list_ast_present_entity[-1] +" "+ word
                        output_text_with_classification = output_text_with_classification + " " + word 
                        # tags: 'O', 'B-problem', 'I-problem', 'I-problem', 'O',
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-present> " 
                    elif pred_assertion[index] == "Possible":
                        list_ast_posssible_entity[-1] = list_ast_posssible_entity[-1] +" "+ word
                        output_text_with_classification = output_text_with_classification + " " + word 
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-possible> " 
                    elif pred_assertion[index] == "Conditional":
                        list_ast_conditional_entity[-1] = list_ast_conditional_entity[-1] +" "+ word
                        output_text_with_classification = output_text_with_classification + " " + word 
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-conditional> " 
                    elif pred_assertion[index] == "Hypothetical":
                        list_ast_hyphothetical_entity[-1] = list_ast_hyphothetical_entity[-1] +" "+ word
                        output_text_with_classification = output_text_with_classification + " " + word 
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-hypothetical> " 
                    elif pred_assertion[index] == "Associated with someone else":
                        list_ast_associated_entity[-1] = list_ast_associated_entity[-1] +" "+ word
                        output_text_with_classification = output_text_with_classification + " " + word 
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-associated> " 
                    elif pred_assertion[index] == "Absent":
                        list_ast_absent_entity[-1] = list_ast_absent_entity[-1] +" "+ word
                        output_text_with_classification = output_text_with_classification + " " + word 
                        if tags[i+1] != 'I-problem':
                            output_text_with_classification = output_text_with_classification + " </Problem-absent> " 
            elif i in treatment_index:
                if (tags[i] == 'B-treatment') : 
                    output_text_with_classification = output_text_with_classification + " <Treatment> " + word
                    # For one word Treatment-entity, there is only 'beginning' tag no 'inside' tag
                    if tags[i+1] != 'I-treatment':
                        output_text_with_classification = output_text_with_classification + " </Treatment> " 
                elif (tags[i] == 'I-treatment') :
                    output_text_with_classification = output_text_with_classification + " " + word 
                    if tags[i+1] != 'I-treatment':
                        output_text_with_classification = output_text_with_classification + " </Treatment> " 
            elif i in test_index:
                if (tags[i] == 'B-test') : 
                    output_text_with_classification = output_text_with_classification + " <Test> " + word
                    # For one word Test-entity, there is only 'beginning' tag no 'inside' tag
                    if tags[i+1] != 'I-test':
                        output_text_with_classification = output_text_with_classification + " </Test> " 
                elif (tags[i] == 'I-test') :
                    output_text_with_classification = output_text_with_classification + " " + word 
                    if tags[i+1] != 'I-test':
                        output_text_with_classification = output_text_with_classification + " </Test> " 
            else:
                if (word.strip() != '[SEP]' and word.strip() != '[CLS]'):
                    output_text_with_classification = output_text_with_classification + " " + word
    

    return render_template(
        "pred_result.html",
        all_sentences=all_sentences,
        assertion_in_sentence=assertion_in_sentence,
        all_problems_in_text=all_problems_in_text_flatten,
        all_treatment_in_text=all_treatment_in_text,
        all_test_in_text=all_test_in_text,
        list_ast_present_entity=list_ast_present_entity,
        list_ast_posssible_entity=list_ast_posssible_entity,
        list_ast_conditional_entity=list_ast_conditional_entity,
        list_ast_hyphothetical_entity=list_ast_hyphothetical_entity,
        list_ast_associated_entity=list_ast_associated_entity,
        list_ast_absent_entity=list_ast_absent_entity,
    )


@app.route("/evaluate", methods=["POST"])
def evaluate():
    # randomly select a file
    f_name = random.choice(os.listdir(config.INDIVIDUAL_TEST))
    f_path = os.path.join(config.INDIVIDUAL_TEST, f_name)
    dataframe = pd.read_csv(f_path, sep="\t").astype(str)
    sentences, labels = get_sentence_label(dataframe)
    input_ids, input_tags, attention_masks = process_data(
        sentences, labels, tokenizer_ner
    )
    # temp_token, _, _ = create_query(sentences, tokenizer)
    query_inputs = torch.tensor(input_ids)
    query_tags = torch.tensor(input_tags)
    query_masks = torch.tensor(attention_masks)
    y_true, y_pred = model_evaluation(query_inputs, query_tags, query_masks, model_ner)
    # Get acc , recall, F1 result report
    report = classification_report(y_true, y_pred, digits=4)
    # Save the report into file
    acc = "{:.2f}".format(accuracy_score(y_true, y_pred))
    f1 = "{:.2f}".format(f1_score(y_true, y_pred))

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

    return render_template(
        "eval_result.html",
        file_name=os.path.basename(f_name),
        sentences=sentences,
        true_label=y_true,
        pred_label=y_pred,
        acc=acc,
        f1=f1,
        column_names=df.columns.values,
        row_data=list(df.values.tolist()),
        zip=zip,
    )

    # return render_template("eval_result.html", sentences=sentences,
    #                        true_label=y_true, pred_label=y_pred,
    #                        file_name=os.path.basename(f_name))


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=8880)

