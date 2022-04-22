import os, re, pickle
import numpy as np
import pandas as pd
import json
from util import config
np.random.seed(1)


def process_concept(concept_str):
    """
    takes string like
    'c="asymptomatic" 16:2 16:2||t="problem"'
    and returns dictionary like
    {'t': 'problem', 'start_line': 16, 'start_pos': 2, 'end_line': 16, 'end_pos': 2}
    """
    try:
        position_bit, problem_bit = concept_str.split('||')
        t = problem_bit[3:-1]
        
        start_and_end_span = next(re.finditer('\s\d+:\d+\s\d+:\d+', concept_str)).span()
        c = concept_str[3:start_and_end_span[0]-1]
        c = [y for y in c.split(' ') if y.strip() != '']
        c = ' '.join(c)

        start_and_end = concept_str[start_and_end_span[0]+1 : start_and_end_span[1]]
        start, end = start_and_end.split(' ')
        start_line, start_pos = [int(x) for x in start.split(':')]
        end_line, end_pos = [int(x) for x in end.split(':')]
        
        # Stupid and hacky!!!! This particular example raised a bug in my code below.
#         if c == 'folate' and start_line == 43 and start_pos == 3 and end_line == 43 and end_pos == 3:
#             start_pos, end_pos = 2, 2
        
    except:
        print(concept_str)
        raise
    
    return {
        't': t, 'start_line': start_line, 'start_pos': start_pos, 'end_line': end_line, 'end_pos': end_pos,
        'c': c, 
    }


"""
Input String: 'c="coronary artery disease" 115:0 115:2||t="problem"||a="present"'
Return Dict: {'t': 'present', 'start_line': 16, 'start_pos': 2, 'end_line': 16, 'end_pos': 2}
"""
def process_assert(ast_str):
    #print(ast_str)
    try:
        position_bit, problem_bit, assertion_bit = ast_str.split('||')
        a = assertion_bit[3:-1]
        
        start_and_end_span = next(re.finditer('\s\d+:\d+\s\d+:\d+', ast_str)).span()
        c = ast_str[3:start_and_end_span[0]-1]
        c = [y for y in c.split(' ') if y.strip() != '']
        c = ' '.join(c)

        start_and_end = ast_str[start_and_end_span[0]+1 : start_and_end_span[1]]
        start, end = start_and_end.split(' ')
        start_line, start_pos = [int(x) for x in start.split(':')]
        end_line, end_pos = [int(x) for x in end.split(':')]
        
    except:
        print(ast_str)
        raise
    
    return {
        'a': a, 'start_line': start_line, 'start_pos': start_pos, 'end_line': end_line, 'end_pos': end_pos,
        'c': c, 
    }



def build_label_vocab(base_dirs):
    seen, label_vocab, label_vocab_size = set(['O']), {'O': 'O'}, 0
    seen_ast, label_vocab_ast, label_vocab_size_ast = set(['O']), {'O': 'O'}, 0
    
    for base_dir in base_dirs:
        concept_dir = os.path.join(base_dir, 'concept')

        assert os.path.isdir(concept_dir), "Directory structure doesn't match!"

        ids = set([x[:-4] for x in os.listdir(concept_dir) if x.endswith('.con')])

        for i in ids:
            with open(os.path.join(concept_dir, '%s.con' % i)) as f:
                concepts = [process_concept(x.strip()) for x in f.readlines()]
            for c in concepts:
                if c['t'] not in seen:
                    label_vocab_size += 1
                    label_vocab['B-%s' % c['t']] = 'B-%s' % c['t'] # label_vocab_size
                    label_vocab_size += 1
                    label_vocab['I-%s' % c['t']] = 'I-%s' % c['t'] # label_vocab_size
                    seen.update([c['t']])


    for base_dir in base_dirs:
        ast_dir = os.path.join(base_dir, 'ast')

        assert os.path.isdir(ast_dir), "Directory structure doesn't match!"

        ids = set([x[:-4] for x in os.listdir(ast_dir) if x.endswith('.ast')])

        for i in ids:
            with open(os.path.join(ast_dir, '%s.ast' % i)) as f:
                assertions = [process_assert(x.strip()) for x in f.readlines()]
            for assertion in assertions:
                if assertion['a'] not in seen_ast:
                    label_vocab_size_ast += 1
                    label_vocab_ast['B-%s' % assertion['a']] = 'B-%s' % assertion['a'] # label_vocab_size
                    label_vocab_size_ast += 1
                    label_vocab_ast['I-%s' % assertion['a']] = 'I-%s' % assertion['a'] # label_vocab_size
                    seen_ast.update([assertion['a']])
    
    return label_vocab, label_vocab_size, label_vocab_ast, label_vocab_size_ast


def reformatter(base, label_vocab, txt_dir = None, concept_dir = None):
    
    
    if txt_dir is None: txt_dir = os.path.join(base, 'txt')
    if concept_dir is None: concept_dir = os.path.join(base, 'concept')
    
    assert os.path.isdir(txt_dir) and os.path.isdir(concept_dir), "Directory structure doesn't match!"
    
    txt_ids = set([x[:-4] for x in os.listdir(txt_dir) if x.endswith('.txt')])
    concept_ids = set([x[:-4] for x in os.listdir(concept_dir) if x.endswith('.con')])
    
    assert txt_ids == concept_ids, (
        "id set doesn't match: txt - concept = %s, concept - txt = %s"
        "" % (str(txt_ids - concept_ids), str(concept_ids - txt_ids))
    )
    
    ids = txt_ids
    
    reprocessed_texts = {}
    for i in ids:
        with open(os.path.join(txt_dir, '%s.txt' % i), mode='r') as f:
            lines = f.readlines()
            txt = [[y for y in x.strip().split(' ') if y.strip() != ''] for x in lines]
            line_starts_with_space = [x.startswith(' ') for x in lines]
        with open(os.path.join(concept_dir, '%s.con' % i), mode='r') as f:
            concepts = [process_concept(x.strip()) for x in f.readlines()]
            
        labels = [['O' for _ in line] for line in txt]
        
        for c in concepts:
            if c['start_line'] == c['end_line']:
                line = c['start_line']-1
                p_modifier = -1 if line_starts_with_space[line] else 0
                text = (' '.join(txt[line][c['start_pos']+p_modifier:c['end_pos']+1+p_modifier])).lower()
                assert text == c['c'], (
                    "Text mismatch! %s vs. %s (id: %s, line: %d)\nFull line: %s"
                    "" % (c['c'], text, i, line, txt[line])
                )
                
            for line in range(c['start_line']-1, c['end_line']):
                p_modifier = -1 if line_starts_with_space[line] else 0
                start_pos = c['start_pos']+p_modifier if line == c['start_line']-1 else 0
                end_pos   = c['end_pos']+1+p_modifier if line == c['end_line']-1 else len(txt[line])
                
                if line == c['end_line'] - 1: labels[line][end_pos-1] = label_vocab['I-%s' % c['t']]                
                if line == c['start_line'] - 1: labels[line][start_pos] = label_vocab['B-%s' % c['t']]
                for j in range(start_pos + 1, end_pos-1): labels[line][j] = label_vocab['I-%s' % c['t']]
            
        joined_words_and_labels = [zip(txt_line, label_line) for txt_line, label_line in zip(txt, labels)]

        #print("joined_words_and_labels :: ", joined_words_and_labels)

        out_str = '\n\n'.join(
            ['\n'.join(['%s %s' % p for p in joined_line]) for joined_line in joined_words_and_labels]
        )
        
        reprocessed_texts[i] = out_str
        
    return reprocessed_texts

def reformatter_with_ast(base, label_vocab, label_vocab_ast, txt_dir = None, concept_dir = None, ast_dir = None):
    
    if txt_dir is None: txt_dir = os.path.join(base, 'txt')
    if concept_dir is None: concept_dir = os.path.join(base, 'concept')
    if ast_dir is None: ast_dir = os.path.join(base, 'ast')
    
    assert os.path.isdir(txt_dir) and os.path.isdir(concept_dir), "Txt and Con directory structure doesn't match!"
    assert os.path.isdir(txt_dir) and os.path.isdir(ast_dir), "Txt and Ast directory structure doesn't match!"
    
    txt_ids = set([x[:-4] for x in os.listdir(txt_dir) if x.endswith('.txt')])
    concept_ids = set([x[:-4] for x in os.listdir(concept_dir) if x.endswith('.con')])
    ast_ids = set([x[:-4] for x in os.listdir(ast_dir) if x.endswith('.ast')])


    assert txt_ids == concept_ids, (
        "id set doesn't match: txt - concept = %s, concept - txt = %s"
        "" % (str(txt_ids - concept_ids), str(concept_ids - txt_ids))
    )

    assert txt_ids == ast_ids, (
        "id set doesn't match: txt - ast = %s, ast - txt = %s"
        "" % (str(txt_ids - ast_ids), str(ast_ids - txt_ids))
    )
    
    ids = txt_ids
    
    reprocessed_texts = {}
    for i in ids:
        with open(os.path.join(txt_dir, '%s.txt' % i), mode='r') as f:
            lines = f.readlines()
            txt = [[y for y in x.strip().split(' ') if y.strip() != ''] for x in lines]
            line_starts_with_space = [x.startswith(' ') for x in lines]

        with open(os.path.join(concept_dir, '%s.con' % i), mode='r') as f:
            concepts = [process_concept(x.strip()) for x in f.readlines()]

        with open(os.path.join(ast_dir, '%s.ast' % i), mode='r') as f:
            assertions = [process_assert(x.strip()) for x in f.readlines()]
            
        labels = [['O' for _ in line] for line in txt]
        for c in concepts:
            if c['start_line'] == c['end_line']:
                line = c['start_line']-1
                p_modifier = -1 if line_starts_with_space[line] else 0
                text = (' '.join(txt[line][c['start_pos']+p_modifier:c['end_pos']+1+p_modifier])).lower()
                assert text == c['c'], (
                    "Text mismatch! %s vs. %s (id: %s, line: %d)\nFull line: %s"
                    "" % (c['c'], text, i, line, txt[line])
                )
                
            for line in range(c['start_line']-1, c['end_line']):
                p_modifier = -1 if line_starts_with_space[line] else 0
                start_pos = c['start_pos']+p_modifier if line == c['start_line']-1 else 0
                end_pos   = c['end_pos']+1+p_modifier if line == c['end_line']-1 else len(txt[line])
                
                if line == c['end_line'] - 1: labels[line][end_pos-1] = label_vocab['I-%s' % c['t']]                
                if line == c['start_line'] - 1: labels[line][start_pos] = label_vocab['B-%s' % c['t']]
                for j in range(start_pos + 1, end_pos-1): labels[line][j] = label_vocab['I-%s' % c['t']]

        # Assertion task
        labels_ast = [['O' for _ in line] for line in txt]
        for c in assertions:
            #print("Assertion :: ", c)
            if c['start_line'] == c['end_line']:
                line = c['start_line']-1
                p_modifier = -1 if line_starts_with_space[line] else 0
                text = (' '.join(txt[line][c['start_pos']+p_modifier:c['end_pos']+1+p_modifier])).lower()
                assert text == c['c'], (
                    "Text mismatch! %s vs. %s (id: %s, line: %d)\nFull line: %s"
                    "" % (c['c'], text, i, line, txt[line])
                )
                
            for line in range(c['start_line']-1, c['end_line']):
                p_modifier = -1 if line_starts_with_space[line] else 0
                start_pos = c['start_pos']+p_modifier if line == c['start_line']-1 else 0
                end_pos   = c['end_pos']+1+p_modifier if line == c['end_line']-1 else len(txt[line])
                
                if line == c['end_line'] - 1: labels_ast[line][end_pos-1] = label_vocab_ast['I-%s' % c['a']]                
                if line == c['start_line'] - 1: labels_ast[line][start_pos] = label_vocab_ast['B-%s' % c['a']]
                for j in range(start_pos + 1, end_pos-1): labels_ast[line][j] = label_vocab_ast['I-%s' % c['a']]
            
        joined_words_and_labels = [zip(txt_line, label_line, label_line_ast) for txt_line, label_line, label_line_ast  in zip(txt, labels, labels_ast)]

        out_str = '\n\n'.join(
            ['\n'.join(['%s %s %s' % p for p in joined_line]) for joined_line in joined_words_and_labels]
        )
        reprocessed_texts[i] = out_str
        
    return reprocessed_texts

def save_to_csv(text, save_path):
    sentence_dict = {}
    all_sentences = [i.split('\n') for i in text.split('\n\n')]

    for i, s in enumerate(all_sentences):
        s = [x for x in s if x]
        if s:
            sentence_dict['sentence# ' + str(i)] = s

    sentence_num_list = list(sentence_dict.keys())
    sentence_list = list(sentence_dict.values())    
    data_tuples = list(zip(sentence_num_list, sentence_list))
    df = pd.DataFrame(data_tuples, columns=['sentence #','sentence'])
    df_new = df.explode(column=['sentence'])
    df_new['word'] = df_new['sentence'].apply(lambda x: x.split()[0])
    df_new['tag'] = df_new['sentence'].apply(lambda x: x.split()[1])
    df_new['ast'] = df_new['sentence'].apply(lambda x: x.split()[2])
    df_new = df_new.drop(columns=['sentence'])
    df_new.to_csv(save_path, index='False', sep='\t')
    

def main():
    label_vocab, label_vocab_size, label_vocab_ast, label_vocab_size_ast = build_label_vocab([
    config.RAW_TRAIN_DIRS['beth'],
    config.RAW_TRAIN_DIRS['partners']
    #'raw/reference_standard_for_test_data/'
    ])
    
    with open(config.OUT_FILES['label'], 'w') as f: f.write('\n'.join(list(label_vocab.keys())))
    with open(config.OUT_FILES['label_assertion'], 'w') as f: f.write('\n'.join(list(label_vocab_ast.keys())))
    
    '''
    reprocessed_texts = {
    'beth':     reformatter(config.RAW_TRAIN_DIRS['beth'], label_vocab),
    'partners': reformatter(config.RAW_TRAIN_DIRS['partners'], label_vocab),
    'test':     reformatter(config.TEST_DIR, label_vocab, txt_dir=config.TEST_TEXT_PATH, concept_dir=config.TEST_CONCEPT_PATH),
    }
    '''
    
    reprocessed_texts = {
    'beth':     reformatter_with_ast(config.RAW_TRAIN_DIRS['beth'], label_vocab, label_vocab_ast),
    'partners': reformatter_with_ast(config.RAW_TRAIN_DIRS['partners'], label_vocab, label_vocab_ast),
    'test':     reformatter_with_ast(config.TEST_DIR, label_vocab, label_vocab_ast, txt_dir=config.TEST_TEXT_PATH,
    concept_dir=config.TEST_CONCEPT_PATH, ast_dir=config.TEST_ASSERTION_PATH),
    }
    
    # How many docs.
    for key, txt_by_record in reprocessed_texts.items(): print("No of docs : %s: %d" % (key, len(txt_by_record)))
    
    # Split training set into training and dev
    all_partners_train_ids = np.random.permutation(list(reprocessed_texts['partners'].keys()))
    N = len(all_partners_train_ids)
    N_train = int(0.9 * N)

    partners_train_ids = all_partners_train_ids[:N_train]
    partners_dev_ids = all_partners_train_ids[N_train:]
    
    all_beth_train_ids = np.random.permutation(list(reprocessed_texts['beth'].keys()))
    N = len(all_beth_train_ids)
    N_train = int(0.9 * N)

    beth_train_ids = all_beth_train_ids[:N_train]
    beth_dev_ids = all_beth_train_ids[N_train:]
    
    merged_train_txt = '\n\n'.join(np.random.permutation(
        [reprocessed_texts['partners'][i] for i in partners_train_ids] + 
        [reprocessed_texts['beth'][i] for i in beth_train_ids]
    ))
    merged_dev_txt = '\n\n'.join(np.random.permutation(
        [reprocessed_texts['partners'][i] for i in partners_dev_ids] + 
        [reprocessed_texts['beth'][i] for i in beth_dev_ids]
    ))
    merged_test_txt = '\n\n'.join(np.random.permutation(list(reprocessed_texts['test'].values())))
    
    print("Merged # Samples: Train: %d, Dev: %d, Test: %d" % (
        len(merged_train_txt.split('\n\n')),
        len(merged_dev_txt.split('\n\n')),
        len(merged_test_txt.split('\n\n'))
    ))
    
    save_to_csv(merged_train_txt, config.OUT_FILES['merged_train'])
    save_to_csv(merged_dev_txt, config.OUT_FILES['merged_dev'])
    save_to_csv(merged_test_txt, config.OUT_FILES['merged_test'])
    with open(config.OUT_FILES['label'], 'w') as f: f.write('\n'.join(list(label_vocab.keys())))

    # with open(config.OUT_FILES['vocab'], mode='wb') as f: json.dump(label_vocab, f)
    # pickle.dumps(label_vocab, config.OUT_FILES['vocab'])
    print('Preprocessing Done.')



if __name__ == "__main__":
    main()