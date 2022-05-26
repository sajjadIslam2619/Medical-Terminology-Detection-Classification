import os
import re
import numpy as np
from util import config


'''
Read data from 'raw' files and chanage traning, test data format into desired format (sentence label).  
'''

def save_to_csv(lines, save_path):
    file = open(save_path, 'w')
    header_line = 'sentence' + '\t' + 'label' + '\n\n'
    file.write(header_line + lines + '\n\n')
    file.close()


def process_assert(ast_str):
    try:
        position_bit, problem_bit, assertion_bit = ast_str.split('||')
        a = assertion_bit[3:-1]

        start_and_end_span = next(re.finditer('\s\d+:\d+\s\d+:\d+', ast_str)).span()
        c = ast_str[3:start_and_end_span[0] - 1]
        c = [y for y in c.split(' ') if y.strip() != '']
        c = ' '.join(c)

        start_and_end = ast_str[start_and_end_span[0] + 1: start_and_end_span[1]]
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


def fetch_label(base_dirs):
    seen_ast, label_vocab_ast, label_vocab_size_ast = set(['O']), {'O': 'O'}, 0
    for base_dir in base_dirs:
        ast_dir = os.path.join(base_dir, 'ast')
        txt_dir = os.path.join(base_dir, 'txt')
        print(ast_dir)
        # ast_dir = os.path.join(base_dir, 'test_ast')
        # txt_dir = os.path.join(base_dir, 'test_txt')

        assert os.path.isdir(ast_dir), "Ast directory structure doesn't match!"
        assert os.path.isdir(txt_dir), "Txt directory structure doesn't match!"

        file_ids = set([x[:-4] for x in os.listdir(ast_dir) if x.endswith('.ast')])

        for file_id in file_ids:
            assertions = []
            with open(os.path.join(ast_dir, '%s.ast' % file_id)) as ast_file:
                assertions = [process_assert(ast_str.strip()) for ast_str in ast_file.readlines()]

            for assertion in assertions:
                if assertion['a'] not in seen_ast:
                    label_vocab_size_ast += 1
                    label_vocab_ast['%s' % assertion['a']] = '%s' % assertion['a']  # label_vocab_size
                    seen_ast.update([assertion['a']])

    return label_vocab_ast, label_vocab_size_ast


def format_data_label(base_dir, txt_dir=None, ast_dir=None):
    line_and_label_list = []

    if txt_dir is None: txt_dir = os.path.join(base_dir, 'txt')
    if ast_dir is None: ast_dir = os.path.join(base_dir, 'ast')
    # ast_dir = os.path.join(base_dir, 'ast')
    # txt_dir = os.path.join(base_dir, 'txt')

    # ast_dir = os.path.join(base_dir, 'test_ast')
    # txt_dir = os.path.join(base_dir, 'test_txt')
    assert os.path.isdir(ast_dir), "Ast directory structure doesn't match!"
    assert os.path.isdir(txt_dir), "Txt directory structure doesn't match!"

    file_ids = set([x[:-4] for x in os.listdir(ast_dir) if x.endswith('.ast')])

    for file_id in file_ids:
        assertions = []
        with open(os.path.join(ast_dir, '%s.ast' % file_id)) as ast_file:
            assertions = [process_assert(ast_str.strip()) for ast_str in ast_file.readlines()]

        with open(os.path.join(txt_dir, '%s.txt' % file_id), mode='r') as txt_file:
            lines = txt_file.readlines()
            txt = [[y for y in x.strip().split(' ') if y.strip() != ''] for x in lines]
            line_starts_with_space = [x.startswith(' ') for x in lines]

        for a in assertions:
            if a['start_line'] == a['end_line']:
                line = a['start_line'] - 1
                p_modifier = -1 if line_starts_with_space[line] else 0
                text = (' '.join(txt[line][a['start_pos'] + p_modifier:a['end_pos'] + 1 + p_modifier])).lower()

                assert text == a['c'], (
                        "Text mismatch! %s vs. %s (id: %s, line: %d)\nFull line: %s"
                        "" % (a['c'], text, file_id, line, txt[line])
                )

            for line_no in range(a['start_line'] - 1, a['end_line']):
                p_modifier = -1 if line_starts_with_space[line_no] else 0
                start_pos = a['start_pos'] + p_modifier if line_no == a['start_line'] - 1 else 0
                end_pos = a['end_pos'] + 1 + p_modifier if line_no == a['end_line'] - 1 else len(txt[line_no])

                # line_format = txt[line_no] # Pass by ref

                line_format = txt[line_no][:]  # Pass by value
                # import copy
                # line_format = copy.deepcopy(txt[line_no])

                line_format.insert(end_pos, '[entity]')
                line_format.insert(start_pos, '[entity]')

                line_and_label = ' '.join(line_format) + '\t' + a['a']
                line_and_label_list.append(line_and_label)

    # print(line_and_label_list)

    return line_and_label_list


def main():
    """
    label_vocab_ast, label_vocab_size_ast = fetch_label([
        config.RAW_TRAIN_DIRS['beth'],
        config.RAW_TRAIN_DIRS['partners']
        # 'raw/reference_standard_for_test_data/'
    ])

    with open(config.OUT_FILES['label_ast'], 'w') as f: f.write('\n'.join(list(label_vocab_ast.keys())))
    """

    reprocessed_texts = {
        'beth': format_data_label(config.RAW_TRAIN_DIRS['beth']),
        'partners': format_data_label(config.RAW_TRAIN_DIRS['partners']),
        'test': format_data_label(config.TEST_DIR, txt_dir=config.TEST_TEXT_PATH, ast_dir=config.TEST_ASSERTION_PATH),
    }

    # How many docs.
    for key, txt_by_record in reprocessed_texts.items(): print("No of docs : %s: %d" % (key, len(txt_by_record)))

    # Split training set into training and dev
    all_partners_train_data = np.random.permutation(reprocessed_texts['partners'])
    N = len(all_partners_train_data)
    N_train = int(0.9 * N)

    partners_train_lines = all_partners_train_data[:N_train]
    partners_dev_lines = all_partners_train_data[N_train:]

    all_beth_train_data = np.random.permutation(reprocessed_texts['beth'])
    N = len(all_beth_train_data)
    N_train = int(0.9 * N)

    beth_train_lines = all_beth_train_data[:N_train]
    beth_dev_lines = all_beth_train_data[N_train:]

    merged_train_txt = '\n\n'.join(np.random.permutation(
        [line for line in partners_train_lines] +
        [line for line in beth_train_lines]
    ))

    merged_dev_txt = '\n\n'.join(np.random.permutation(
        [line for line in partners_dev_lines] +
        [line for line in beth_dev_lines]
    ))

    merged_test_txt = '\n\n'.join(np.random.permutation(reprocessed_texts['test']))

    save_to_csv(merged_train_txt, config.OUT_FILES['assertion_label_train'])
    save_to_csv(merged_dev_txt, config.OUT_FILES['assertion_label_dev'])
    save_to_csv(merged_test_txt, config.OUT_FILES['assertion_label_test'])

    return


if __name__ == "__main__":
    main()
