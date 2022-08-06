from flask import Flask, render_template, request, flash
from flask import send_file, url_for, jsonify

import jinja2

env = jinja2.Environment()
env.globals.update(zip=zip)
import os
import random
from werkzeug.utils import secure_filename
import copy
import sys
sys.path.append('NER')
from NER.processor import *
from NER.ner_utils import *
from NER import config
from Assertion.assertion_utils import *
from utils import *
from seqeval.metrics import classification_report, accuracy_score, f1_score
import stanza
from zipfile import ZipFile
import json
#import logging, sys

#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

try:
    nlp = stanza.Pipeline(lang="en", processors="tokenize")
except Exception:
    stanza.download("en")
    nlp = stanza.Pipeline(lang="en", processors="tokenize")

model_ner, tokenizer_ner = load_ner_model()
model_assertion, tokenizer_assertion = load_assertion_model()
output_eval_file = "./Results/eval_results.txt"
output_classification_file = "static/download/txt/output_classification.txt"
output_file_path = "static/download/batch-process/"
output_JSON_path = "static/download/JSON/output_entity.json"
output_zip_name = "output_files.zip"

UPLOAD_FOLDER = "./Results"
UPLOAD_FILES_FOLDER = "./Static/upload/txt"
ALLOWED_EXTENSIONS = {"txt"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["UPLOAD_FILES_FOLDER"] = UPLOAD_FILES_FOLDER


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


# This method add start-tag if there is no Begin (B) before Inside (I) and Unseen (X). 
# Ast : - Present   Present  - Present   Absent    Absent    -
# Tag : O I-problem I-roblem O I-problem I-problem I-problem O
# output : O <present> B-problem </present> <present> B-problem X I-problem </present> <absent> B-problem I-problem </absent> O 

def add_problem_start_tag(
    output_text_with_classification,
    word,
    i,
    tags,
    assertion_index,
    pred_assertion,
    list_ast_entity,
    assertion,
    tag_string,
):
    if i-1 not in assertion_index:
        # Ast   : -    P   -     P    -
        # Tag   : O <> I I O <> I I I O
        list_ast_entity.append(word)
        output_text_with_classification = (
            output_text_with_classification + tag_string + word
        )
    else: 
        prev_index = assertion_index.index(i)
        if pred_assertion[prev_index] != assertion:
            # Ast   : -     - P     A   -
            # Tag   : O I I O I <>  I I O
            list_ast_entity.append(word)
            output_text_with_classification = (
                output_text_with_classification + tag_string + word
            )
        else :
            list_ast_entity[-1] = (
                list_ast_entity[-1] + " " + word
            )
            output_text_with_classification = (
                output_text_with_classification + " " + word
            )

    return output_text_with_classification, list_ast_entity


# This method add end-tag for all cases. 
# Ast : - Present   Present  Present Present   Absent    Absent    -
# Tag : O B-problem B-roblem X       I-problem B-problem I-problem O
# output : O <present> B-problem </present> <present> B-problem X I-problem </present> <absent> B-problem I-problem </absent> O 

def add_problem_end_tag(
    output_text_with_classification,
    i,
    tags,
    assertion_index,
    pred_assertion,
    assertion,
    tag_string,
):
    if i + 1 in assertion_index:
        next_index = assertion_index.index(i + 1)
        if pred_assertion[next_index] == assertion:
            # Ast :  P     P
            # Tag: O B </> B X I B I O
            if tags[i+1] == "B-problem": 
                output_text_with_classification = (
                    output_text_with_classification + tag_string
                )
        else :
            # Ast :        P     A 
            # Tag: O B B X I </> B I O
            output_text_with_classification = (
                output_text_with_classification + tag_string
            )
            
    else:
        # Ast :             A
        # Tag : O B B X I B I </> O
        output_text_with_classification = output_text_with_classification + tag_string

    return output_text_with_classification


def process_sentence(
    sentence,
    tags,
    pred_assertion,
    assertion_index,
    treatment_index,
    test_index,
    output_text_with_classification,
    list_ast_present_entity,
    list_ast_absent_entity,
    list_ast_posssible_entity,
    list_ast_conditional_entity,
    list_ast_hyphothetical_entity,
    list_ast_associated_entity,
    list_treatment_entity,
    list_test_entity,
):
    #print('sentence ::: ', sentence)
    #print('tags ::: ', tags)
    #print('pred_assertion ::: ', pred_assertion)
    #print('assertion_index ::: ', assertion_index)
    #print('treatment_index ::: ', treatment_index)
    #print('test_index ::: ', test_index)

    for i, word in enumerate(sentence):

        if i in assertion_index:
            index = assertion_index.index(i)
            if tags[i] == "B-problem":
                if pred_assertion[index] == "Present":
                    list_ast_present_entity.append(word)
                    output_text_with_classification = (
                        output_text_with_classification + " <Problem-present> " + word
                    )
                    # For one word problem-entity, there is only 'beginning' tag no 'inside' tag
                    # tags : 'O', 'O', 'B-problem', 'O', 'O',
                    output_text_with_classification = add_problem_end_tag(
                        output_text_with_classification,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        "Present",
                        " </Problem-present> ",
                    )

                elif pred_assertion[index] == "Possible":
                    list_ast_posssible_entity.append(word)
                    output_text_with_classification = (
                        output_text_with_classification + " <Problem-possible> " + word
                    )
                    output_text_with_classification = add_problem_end_tag(
                        output_text_with_classification,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        "Possible",
                        " </Problem-possible> ",
                    )

                elif pred_assertion[index] == "Conditional":
                    list_ast_conditional_entity.append(word)
                    output_text_with_classification = (
                        output_text_with_classification
                        + " <Problem-conditional> "
                        + word
                    )
                    output_text_with_classification = add_problem_end_tag(
                        output_text_with_classification,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        "Conditional",
                        " </Problem-conditional> ",
                    )

                elif pred_assertion[index] == "Hypothetical":
                    list_ast_hyphothetical_entity.append(word)
                    output_text_with_classification = (
                        output_text_with_classification
                        + " <Problem-hypothetical> "
                        + word
                    )
                    output_text_with_classification = add_problem_end_tag(
                        output_text_with_classification,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        "Hypothetical",
                        " </Problem-hypothetical> ",
                    )

                elif pred_assertion[index] == "Associated with someone else":
                    list_ast_associated_entity.append(word)
                    output_text_with_classification = (
                        output_text_with_classification
                        + " <Problem-associated> "
                        + word
                    )
                    output_text_with_classification = add_problem_end_tag(
                        output_text_with_classification,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        "Associated with someone else",
                        " </Problem-associated> ",
                    )

                elif pred_assertion[index] == "Absent":
                    list_ast_absent_entity.append(word)
                    output_text_with_classification = (
                        output_text_with_classification + " <Problem-absent> " + word
                    )
                    output_text_with_classification = add_problem_end_tag(
                        output_text_with_classification,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        "Absent",
                        " </Problem-absent> ",
                    )

            elif tags[i] == "I-problem" or tags[i] == "X":
                if pred_assertion[index] == "Present":
                    (output_text_with_classification, list_ast_present_entity) = add_problem_start_tag(
                        output_text_with_classification,
                        word,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        list_ast_present_entity,
                        "Present",
                        " <Problem-present> ",
                    )

                    output_text_with_classification = add_problem_end_tag(
                        output_text_with_classification,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        "Present",
                        " </Problem-present> ",
                    )

                elif pred_assertion[index] == "Possible":

                    (output_text_with_classification, list_ast_posssible_entity) = add_problem_start_tag(
                        output_text_with_classification,
                        word,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        list_ast_posssible_entity,
                        "Possible",
                        " <Problem-possible> ",
                    )
                    output_text_with_classification = add_problem_end_tag(
                        output_text_with_classification,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        "Possible",
                        " </Problem-possible> ",
                    )

                elif pred_assertion[index] == "Conditional":

                    (output_text_with_classification, list_ast_conditional_entity) = add_problem_start_tag(
                        output_text_with_classification,
                        word,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        list_ast_conditional_entity,
                        "Conditional",
                        " <Problem-conditional> ",
                    )
                    output_text_with_classification = add_problem_end_tag(
                        output_text_with_classification,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        "Conditional",
                        " </Problem-conditional> ",
                    )

                elif pred_assertion[index] == "Hypothetical":

                    (output_text_with_classification, list_ast_hyphothetical_entity) = add_problem_start_tag(
                        output_text_with_classification,
                        word,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        list_ast_hyphothetical_entity,
                        "Hypothetical",
                        " <Problem-hypothetical> ",
                    )
                    output_text_with_classification = add_problem_end_tag(
                        output_text_with_classification,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        "Hypothetical",
                        " </Problem-hypothetical> ",
                    )

                elif pred_assertion[index] == "Associated with someone else":

                    (output_text_with_classification, list_ast_associated_entity) = add_problem_start_tag(
                        output_text_with_classification,
                        word,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        list_ast_associated_entity,
                        "Associated with someone else",
                        " <Problem-associated> ",
                    )
                    output_text_with_classification = add_problem_end_tag(
                        output_text_with_classification,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        "Associated with someone else",
                        " </Problem-associated> ",
                    )

                elif pred_assertion[index] == "Absent":

                    (output_text_with_classification, list_ast_absent_entity) = add_problem_start_tag(
                        output_text_with_classification,
                        word,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        list_ast_absent_entity,
                        "Absent",
                        " <Problem-absent> ",
                    )
                    output_text_with_classification = add_problem_end_tag(
                        output_text_with_classification,
                        i,
                        tags,
                        assertion_index,
                        pred_assertion,
                        "Absent",
                        " </Problem-absent> ",
                    )

        elif i in treatment_index:
            if tags[i] == "B-treatment":
                list_treatment_entity.append(word)
                output_text_with_classification = (
                    output_text_with_classification + " <Treatment> " + word
                )
                # For one word Treatment-entity, there is only 'beginning' tag no 'inside' tag
                if tags[i + 1] != "I-treatment" and tags[i + 1] != "X":
                    output_text_with_classification = (
                        output_text_with_classification + " </Treatment> "
                    )
            elif tags[i] == "I-treatment" or tags[i] == "X":
                if i-1 not in treatment_index:
                    # Tag: O I-treatment O I-treatment X 0 
                    list_treatment_entity.append(word)
                    output_text_with_classification = (
                        output_text_with_classification + " <Treatment> " + word
                    )
                else : 
                    list_treatment_entity[-1] = list_treatment_entity[-1] + " " + word
                    output_text_with_classification = (
                        output_text_with_classification + " " + word
                    )

                if tags[i + 1] != "I-treatment" and tags[i + 1] != "X":
                    output_text_with_classification = (
                        output_text_with_classification + " </Treatment> "
                    )
        elif i in test_index:
            if tags[i] == "B-test":
                list_test_entity.append(word)
                output_text_with_classification = (
                    output_text_with_classification + " <Test> " + word
                )
                # For one word Test-entity, there is only 'beginning' tag no 'inside' tag
                if tags[i + 1] != "I-test" and tags[i + 1] != "X":
                    output_text_with_classification = (
                        output_text_with_classification + " </Test> "
                    )
            elif tags[i] == "I-test" or tags[i] == "X":
                if i-1 not in test_index:
                    # Tag: O I-test O I-test X 0 
                    list_test_entity.append(word)
                    output_text_with_classification = (
                        output_text_with_classification + " <Test> " + word
                    )
                else :
                    list_test_entity[-1] = list_test_entity[-1] + " " + word
                    output_text_with_classification = (
                        output_text_with_classification + " " + word
                    )
                if tags[i + 1] != "I-test" and tags[i + 1] != "X":
                    output_text_with_classification = (
                        output_text_with_classification + " </Test> "
                    )
        else:
            if word.strip() != "[SEP]" and word.strip() != "[CLS]":
                output_text_with_classification = (
                    output_text_with_classification + " " + word
                )

    return (
        output_text_with_classification,
        list_ast_present_entity,
        list_ast_absent_entity,
        list_ast_posssible_entity,
        list_ast_conditional_entity,
        list_ast_hyphothetical_entity,
        list_ast_associated_entity,
        list_treatment_entity,
        list_test_entity,
    )


@app.route("/")
def home():
    total_file = 0
    files_name_str = ""
    return render_template(
        "home.html", total_file=total_file, files_name_str=files_name_str
    )


@app.route("/predict_files_batch", methods=["GET", "POST"])
def predict_files_batch():
    all_sentences, all_tags = [], []
    # if request.method != "POST":
    # return None
    # check if the post request has the file part
    if "file" not in request.files:
        flash("No file part")
        return None
    files = request.files.getlist("file")

    # output_format=request.args.get('output_format', None)
    # total_file=request.args.get('total_file', None)
    # files_name_str=request.args.get('files_name_str', None)

    total_file = 0
    files_name_str = ""
    JSON_Obj_dict_list = []

    for i, file in enumerate(files):
        if file.filename == "":
            flash("No selected file")
            return None
        if file and allowed_file(file.filename):
            file_name = secure_filename(file.filename)
            if i > 0:
                files_name_str += ", "
            files_name_str += file_name
            file.save(os.path.join(app.config["UPLOAD_FILES_FOLDER"], file_name))
            file.seek(0)
            long_text = file.read()
            long_text = str(long_text, "utf-8")
            total_file = total_file + 1

            # file_name = file_name.strip()
            file_path = UPLOAD_FILES_FOLDER + "/" + file_name
            file = open(file_path, "r")
            long_text = file.read()

            all_sentences, all_tags = predict_entities(long_text)

            (
                assertion_in_sentence,
                all_problems_in_text_flatten,
                all_treatment_in_text,
                all_test_in_text,
            ) = predict_assertion(all_sentences, all_tags)

            #logging.debug('File Name ::: ', file_name)
            #logging.info('all_sentences ::: ', all_sentences)
            #logging.info('all_tags ::: ', all_tags)

            # These lists are to collect problem-entity label-wise and display on the UI Table and prepare JSON object
            list_ast_present_entity = []
            list_ast_absent_entity = []
            list_ast_posssible_entity = []
            list_ast_conditional_entity = []
            list_ast_hyphothetical_entity = []
            list_ast_associated_entity = []

            list_treatment_entity = []
            list_test_entity = []

            ##### JSON Object structure #####
            # file_name : file_name
            # test : list_test_entity
            # treatment : list_treatment_entity
            # problem : { present: list_ast_present_entity, absent: list_ast_absent_entity, possible : list_ast_posssible_entity ... etc}
            ##### JSON Object structure #####
            JSON_Obj_dict = {}
            JSON_Obj_dict["file_name"] = file_name
            print("file name :: ", file_name)
            # This string is to format full clinical text and add <tag>, like  <Problem-present> problem-entity </Problem-present>
            output_text_with_classification = ""

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
                (
                    output_text_with_classification,
                    list_ast_present_entity,
                    list_ast_absent_entity,
                    list_ast_posssible_entity,
                    list_ast_conditional_entity,
                    list_ast_hyphothetical_entity,
                    list_ast_associated_entity,
                    list_treatment_entity,
                    list_test_entity,
                ) = process_sentence(
                    sentence,
                    tags,
                    pred_assertion,
                    assertion_index,
                    treatment_index,
                    test_index,
                    output_text_with_classification,
                    list_ast_present_entity,
                    list_ast_absent_entity,
                    list_ast_posssible_entity,
                    list_ast_conditional_entity,
                    list_ast_hyphothetical_entity,
                    list_ast_associated_entity,
                    list_treatment_entity,
                    list_test_entity,
                )

            JSON_Obj_Problem_dict = {}
            JSON_Obj_Problem_dict["present"] = list_ast_present_entity
            JSON_Obj_Problem_dict["absent"] = list_ast_absent_entity
            JSON_Obj_Problem_dict["possible"] = list_ast_posssible_entity
            JSON_Obj_Problem_dict["conditional"] = list_ast_conditional_entity
            JSON_Obj_Problem_dict["hypothetical"] = list_ast_hyphothetical_entity
            JSON_Obj_Problem_dict["associated"] = list_ast_associated_entity

            JSON_Obj_dict["problem"] = JSON_Obj_Problem_dict
            JSON_Obj_dict["treatment"] = list_treatment_entity
            JSON_Obj_dict["test"] = list_test_entity

            JSON_Obj_dict_list.append(JSON_Obj_dict)

            output_file = output_file_path + "output_" + file_name
            try:
                with open(output_file, "w") as f:
                    f.write(output_text_with_classification)
            except FileNotFoundError:
                print("The 'static/download/batch-process/' directory does not exist")

            if os.path.exists(
                os.path.join(app.config["UPLOAD_FILES_FOLDER"], file_name)
            ):
                os.remove(os.path.join(app.config["UPLOAD_FILES_FOLDER"], file_name))
            else:
                print(" The file does not exist : ", file_name)

    json_object = json.dumps(JSON_Obj_dict_list, indent=4)
    with open(output_JSON_path, "w") as outfile:
        outfile.write(json_object)

    return render_template(
        "home.html", total_file=total_file, files_name_str=files_name_str
    )


@app.route("/download_files_batch", methods=["GET", "POST"])
def download_files_batch():
    output_format = request.args.get("output_format", None)
    total_file = request.args.get("total_file", None)
    files_name_str = request.args.get("files_name_str", None)
    file_name_list = files_name_str.split(",")

    if output_format == "txt":
        target = output_file_path
        stream = BytesIO()
        with ZipFile(stream, "w") as zf:
            for file in file_name_list:
                file = file.strip()
                file = os.path.join(target, "output_" + file)
                zf.write(file, os.path.basename(file))

                if os.path.exists(file):
                    os.remove(file)
                else:
                    print(" The file does not exist : ", file)
        stream.seek(0)

        return send_file(
            stream, as_attachment=True, attachment_filename=output_zip_name
        )

    elif output_format == "JSON":
        path = output_JSON_path
        return send_file(path, as_attachment=True)

    total_file = 0
    files_name_str = ""
    return render_template(
        "home.html", total_file=total_file, files_name_str=files_name_str
    )


@app.route("/predict_file", methods=["POST"])
def predict_file():
    all_sentences, all_tags = [], []

    # These lists are to collect problem-entity label-wise and display on the UI Table
    list_ast_present_entity = []
    list_ast_absent_entity = []
    list_ast_posssible_entity = []
    list_ast_conditional_entity = []
    list_ast_hyphothetical_entity = []
    list_ast_associated_entity = []

    list_treatment_entity = []
    list_test_entity = []

    # This string is to format full clinical text and add <tag>, like  <Problem-present> problem-entity </Problem-present>
    output_text_with_classification = ""

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

        upload_file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if os.path.exists(upload_file_path):
            os.remove(upload_file_path)
        else:
            print(" The file does not exist : ", upload_file_path)

        all_sentences, all_tags = predict_entities(long_text)

        (
            assertion_in_sentence,
            all_problems_in_text_flatten,
            all_treatment_in_text,
            all_test_in_text,
        ) = predict_assertion(all_sentences, all_tags)

    #logging.debug('File Name ::: ', filename)
    #logging.info('all_sentences ::: ', all_sentences)
    #logging.info('all_tags ::: ', all_tags)
    print("file name", filename)
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
        (
            output_text_with_classification,
            list_ast_present_entity,
            list_ast_absent_entity,
            list_ast_posssible_entity,
            list_ast_conditional_entity,
            list_ast_hyphothetical_entity,
            list_ast_associated_entity,
            list_treatment_entity,
            list_test_entity,
        ) = process_sentence(
            sentence,
            tags,
            pred_assertion,
            assertion_index,
            treatment_index,
            test_index,
            output_text_with_classification,
            list_ast_present_entity,
            list_ast_absent_entity,
            list_ast_posssible_entity,
            list_ast_conditional_entity,
            list_ast_hyphothetical_entity,
            list_ast_associated_entity,
            list_treatment_entity,
            list_test_entity,
        )

    try:
        with open(output_classification_file, "w") as f:
            f.write(output_text_with_classification)
    except FileNotFoundError:
        print("The 'static/download/txt/' directory does not exist")

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
    # These lists are to collect problem-entity label-wise and display on the UI Table
    list_ast_present_entity = []
    list_ast_absent_entity = []
    list_ast_posssible_entity = []
    list_ast_conditional_entity = []
    list_ast_hyphothetical_entity = []
    list_ast_associated_entity = []

    list_treatment_entity = []
    list_test_entity = []

    # This string is to format full clinical text and add <tag>, like  <Problem-present> problem-entity </Problem-present>
    output_text_with_classification = ""

    all_sentences, all_tags = predict_entities(message)
    (
        assertion_in_sentence,
        all_problems_in_text_flatten,
        all_treatment_in_text,
        all_test_in_text,
    ) = predict_assertion(all_sentences, all_tags)

    #logging.debug('all_sentences ::: ', all_sentences)
    #logging.debug('all_tags ::: ', all_tags)

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
        (
            output_text_with_classification,
            list_ast_present_entity,
            list_ast_absent_entity,
            list_ast_posssible_entity,
            list_ast_conditional_entity,
            list_ast_hyphothetical_entity,
            list_ast_associated_entity,
            list_treatment_entity,
            list_test_entity,
        ) = process_sentence(
            sentence,
            tags,
            pred_assertion,
            assertion_index,
            treatment_index,
            test_index,
            output_text_with_classification,
            list_ast_present_entity,
            list_ast_absent_entity,
            list_ast_posssible_entity,
            list_ast_conditional_entity,
            list_ast_hyphothetical_entity,
            list_ast_associated_entity,
            list_treatment_entity,
            list_test_entity,
        )

    try:
        with open(output_classification_file, "w") as f:
            f.write(output_text_with_classification)
    except FileNotFoundError:
        print("The 'static/download/txt/' directory does not exist")

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


@app.route("/download")
def download():
    path = output_classification_file
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=8880)

