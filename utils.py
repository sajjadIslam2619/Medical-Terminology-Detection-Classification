
from NER.processor import *
from NER.ner_utils import *
from NER import config
import copy
from transformers import BertForTokenClassification, BertTokenizer
from transformers import AutoConfig, AutoTokenizer, BertForSequenceClassification


def load_ner_model():
    num_labels = len(tag2idx)
    save_model_address = './trained_models/NER/C-Bert-test'
    model = BertForTokenClassification.from_pretrained(
        save_model_address, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(
        save_model_address, do_lower_case=False)
    return model, tokenizer


def load_assertion_model():
    # no of classifier: present, not-present
    num_labels = 3
    MODEL_CLASSES = {
        'bert': (AutoConfig, BertForSequenceClassification, AutoTokenizer),
    }
    MODEL_ADDRESS = 'emilyalsentzer/Bio_ClinicalBERT'
    config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
    model_config = config_class.from_pretrained(
        MODEL_ADDRESS, num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(
        MODEL_ADDRESS, do_lower_case=False)
    model = model_class.from_pretrained(MODEL_ADDRESS, config=model_config)
    output_dir = './trained_models/Assertion/6_label_model_oversampling'
    model = model_class.from_pretrained(output_dir)
    tokenizer = tokenizer_class.from_pretrained(output_dir)
    return model, tokenizer


# extract index of entity
# extract sentence with assertion
def entity_extractor(all_sentences, all_tags):
    sentences_with_problem = []
    all_problems_in_text_tmp = []
    all_treatment_in_text = []
    all_test_in_text = []

    for s, t in zip(all_sentences, all_tags):
        flag_treatment, flag_problem, flag_test = 0, 0, 0
        problem_in_sentence = ''
        treatment_in_sentence = []
        test_in_sentence = []
        for i in range(1, len(t)-1):
            if t[i] == 'B-problem':
                flag_problem = 1
                # if there is entities, add the index of sentence to a list
                # sentences_with_problem.append(n)
                # append the index of entity to a list
                if problem_in_sentence:
                    problem_in_sentence = f'{problem_in_sentence}| {str(i)}'
                else:
                    problem_in_sentence += str(i)
            elif t[i] == 'I-problem' or t[i] == 'X' and flag_problem == 1:
                problem_in_sentence = f'{problem_in_sentence} {str(i)}'
            elif t[i] == 'B-test':
                flag_test = 1
                test_in_sentence.append(i)
            elif t[i] == 'I-test' or t[i] == 'X' and flag_test == 1:
                test_in_sentence.append(i)
            elif t[i] == 'B-treatment':
                flag_treatment = 1
                treatment_in_sentence.append(i)
            elif t[i] == 'I-treatment' or t[i] == 'X' and flag_treatment == 1:
                treatment_in_sentence.append(i)
            elif t[i] in ['O', 'X']:
                flag_treatment, flag_problem, flag_test = 0, 0, 0
                # print(s[i], end=' ')
        all_problems_in_text_tmp.append(problem_in_sentence)
        all_treatment_in_text.append(treatment_in_sentence)
        all_test_in_text.append(test_in_sentence)

    # create sentences with '[entity]' tag
    all_problems_in_text = []
    sentences_with_problem = []
    for sentence, problem_index in zip(all_sentences, all_problems_in_text_tmp):
        # print(problem_index)
        if problem_index:
            index = problem_index.split('|')
            tmp = [i.split() for i in index]
            all_problems_in_text.append(tmp)
            for i_list in tmp:
                s = copy.deepcopy(sentence)
                s.insert(int(i_list[-1])+1, '[entity]')
                s.insert(int(i_list[0]), '[entity]')
                s = ' '.join(s)
                sentences_with_problem.append(s)
        else:
            # sentences_with_problem.append(sentence)
            all_problems_in_text.append(problem_index)

    return sentences_with_problem, all_problems_in_text, all_treatment_in_text, all_test_in_text
