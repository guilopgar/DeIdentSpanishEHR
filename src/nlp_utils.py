### Module containing auxiliary functions and classes for NLP using Transformers


## Load text

import os


def load_text_files(file_names, path):
    """
    It loads the text contained in a set of files into a returned list of strings.
    Code adapted from https://stackoverflow.com/questions/33912773/python-read-txt-files-into-a-dataframe
    """
    output = []
    for f in file_names:
        with open(path + f, "r") as file:
            output.append(file.read())

    return output


def load_ss_files(file_names, path):
    """
    It loads the start-end pair of each split sentence from a set of files (start + \t + end line-format expected) into a
    returned dictionary, where keys are file names and values a list of tuples containing the start-end pairs of the
    split sentences.
    """
    output = dict()
    for f in file_names:
        with open(path + f, "r") as file:
            f_key = f.split('.')[0]
            output[f_key] = []
            for sent in file:
                output[f_key].append(tuple(map(int, sent.strip().split('\t'))))

    return output


import numpy as np
import pandas as pd


def process_de_ident_ner(brat_files):
    """
    Primarly dessign to process de-identification annotations from Galén corpus.
    brat_files: list containing the path of the annotations files in BRAT format (.ann).
    """

    df_res = []
    for file in brat_files:
        with open(file) as ann_file:
            doc_name = file.split('/')[-1].split('.ann')[0]
            for line in ann_file:
                if line.strip():
                    line_split = line.strip().split('\t')
                    if line_split[0][0] == "T":
                        assert len(line_split) == 3
                        text_ref = line_split[2]
                        ann_type = line_split[1].split(' ')[0]
                        location = ' '.join(line_split[1].split(' ')[1:])
                        df_res.append([doc_name, text_ref, ann_type, location])

    return pd.DataFrame(df_res, columns=["doc_id", "text_ref", "type", "location"])


## Whitespace-punctuation tokenization (same as BERT pre-tokenization)
# The next code is adapted from: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py

import unicodedata


def is_punctuation(ch):
    code = ord(ch)
    return 33 <= code <= 47 or \
        58 <= code <= 64 or \
        91 <= code <= 96 or \
        123 <= code <= 126 or \
        unicodedata.category(ch).startswith('P')


def is_cjk_character(ch):
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or \
        0x3400 <= code <= 0x4DBF or \
        0x20000 <= code <= 0x2A6DF or \
        0x2A700 <= code <= 0x2B73F or \
        0x2B740 <= code <= 0x2B81F or \
        0x2B820 <= code <= 0x2CEAF or \
        0xF900 <= code <= 0xFAFF or \
        0x2F800 <= code <= 0x2FA1F


def is_space(ch):
    return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
        unicodedata.category(ch) == 'Zs'


def is_control(ch):
    """
    Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py#L64
    """
    return unicodedata.category(ch).startswith("C")


def word_start_end(text, start_i=0, cased=True):
    """
    Our aim is to produce both a list of strings containing the text of each word and a list of pairs containing the start and
    end char positions of each word.

    start_i: the start position of the first character in the text.

    Code adapted from: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py#L101
    """

    if not cased:
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        text = text.lower()
    spaced = ''
    # Store the start positions of each considered character (ch) in start_arr,
    # such that sum([len(word) for word in spaced.strip().split()]) = len(start_arr)
    start_arr = []
    for ch in text:
        if is_punctuation(ch) or is_cjk_character(ch):
            spaced += ' ' + ch + ' '
            start_arr.append(start_i)
        elif is_space(ch):
            spaced += ' '
        elif not (ord(ch) == 0 or ord(ch) == 0xfffd or is_control(ch)):
            spaced += ch
            start_arr.append(start_i)
        # If it is a control char we skip it but take its offset into account
        start_i += 1

    assert sum([len(word) for word in spaced.strip().split()]) == len(start_arr)

    text_arr, start_end_arr = [], []
    i = 0
    for word in spaced.strip().split():
        text_arr.append(word)
        j = i + len(word)
        start_end_arr.append((start_arr[i], start_arr[j - 1] + 1))
        i = j

    return text_arr, start_end_arr


## NER-annotations

def start_end_tokenize(text, tokenizer, start_pos=0):
    """
    Our aim is to produce both a list of sub-tokens and a list of tuples containing the start and
    end char positions of each sub-token.
    """
    type_tokenizer = str(type(tokenizer))
    if 'transformers' in type_tokenizer:
        start_end_arr = []
        token_text = tokenizer(text, add_special_tokens=False)
        for i in range(len(token_text['input_ids'])):
            chr_span = token_text.token_to_chars(i)
            start_end_arr.append((chr_span.start + start_pos, chr_span.end + start_pos))

        return tokenizer.convert_ids_to_tokens(token_text['input_ids']), start_end_arr

    elif 'fasttext' in type_tokenizer:
        return [text], [(start_pos, start_pos + len(text))]


# Creation of a NER corpus

def ner_iob2_annotate(arr_start_end, df_ann, subtask='ner'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following either IOB-2 NER format
    (subtask='ner') or IOB-Code NER-Norm format (subtask='norm'), using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.

    Implemented subtasks: ner, norm

    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """

    labels = ["O"] * len(arr_start_end)
    for index, row in df_ann.iterrows():
        # First subtoken/word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= row['start'])[0][-1]  # last subtoken/word <= annotation start
        # Last subtoken/word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= row['end'])[0][0]  # first subtoken/word >= annotation end
        assert tok_start <= tok_end
        # Annotate first subtoken/word
        # assert labels[tok_start] == "O" # no overlapping annotations are expected
        # Because the presence of two ann in a single word, e.g. "pT3N2Mx" ann in Cantemist dev-set2 cc_onco1427
        if labels[tok_start] != "O":
            print(labels[tok_start])
            print(row)
            print(tok_start)
            print(tok_end)
            print(arr_start_end)

        if subtask == 'ner':
            labels[tok_start] = "B"
        elif subtask == 'norm':
            labels[tok_start] = "B" + "-" + row['code']
        else:
            raise Exception('Subtask not implemented!')

        if tok_start < tok_end:
            # Annotation spanning multiple subtokens/words
            for i in range(tok_start + 1, tok_end + 1):
                assert labels[i] == "O"  # no overlapping annotations are expected
                if subtask == 'ner':
                    labels[i] = "I"
                elif subtask == 'norm':
                    labels[i] = "I" + "-" + row['code']

    return [labels]


def norm_iob2_code_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_code'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2, Code] NER-Norm format,
    using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.

    Implemented subtasks: norm-iob_code, norm-iob_code-crf

    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """

    iob_labels = ["O"] * len(arr_start_end)
    default_code_value = "O" if ign_value is None else ign_value
    if subtask.split('-')[-1] != "crf": code_labels = [default_code_value] * len(arr_start_end)
    for index, row in df_ann.iterrows():
        # First subtoken/word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= row['start'])[0][-1]  # last subtoken/word <= annotation start
        # Last subtoken/word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= row['end'])[0][0]  # first subtoken/word >= annotation end
        assert tok_start <= tok_end
        # Annotate first subtoken/word
        # assert labels[tok_start] == "O" # no overlapping annotations are expected
        if iob_labels[
            tok_start] != "O":  # Because of the "pT3N2Mx" annotation (two ann in a single word) in dev-set2 cc_onco1427
            print(iob_labels[tok_start])
            print(row)
            print(tok_start)
            print(tok_end)
            print(arr_start_end)

        iob_labels[tok_start] = "B"
        if subtask.split('-')[-1] == "crf":
            iob_labels[tok_start] += ('-' + row["code"])
        else:
            code_labels[tok_start] = row["code"]

        if tok_start < tok_end:
            # Annotation spanning multiple subtokens/words
            for i in range(tok_start + 1, tok_end + 1):
                assert iob_labels[i] == "O"  # no overlapping annotations are expected
                iob_labels[i] = "I"
                if subtask.split('-')[-1] == "crf":
                    iob_labels[i] += ('-' + row["code"])
                else:
                    code_labels[i] = row["code"]

    if subtask.split('-')[-1] == "crf":
        return [iob_labels]

    else:
        return [iob_labels, code_labels]


def ner_iob2_disc_annotate(arr_start_end, df_ann, subtask='ner'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (disc)]
    NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.

    Implemented subtasks: norm-iob_disc
    """

    iob_disc_labels = ["O"] * len(arr_start_end)
    for index, row in df_ann.iterrows():
        ann_loc_split = row['location'].split(';')
        ## First fragment
        loc_start = int(ann_loc_split[0].split(' ')[0])
        loc_end = int(ann_loc_split[0].split(' ')[1])
        # First subtoken/word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= loc_start)[0][-1]  # last subtoken/word <= annotation start
        # Last subtoken/word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= loc_end)[0][0]  # first subtoken/word >= annotation end
        assert tok_start <= tok_end
        # Annotate first subtoken/word
        assert iob_disc_labels[tok_start] == "O"  # no overlapping annotations are expected
        if subtask == 'ner':
            iob_disc_labels[tok_start] = "B"
        elif subtask == 'norm':
            iob_disc_labels[tok_start] = "B" + "-" + row['code']
        else:
            raise Exception('Subtask not implemented!')

        if tok_start < tok_end:
            # Annotation spanning multiple subtokens/words
            for i in range(tok_start + 1, tok_end + 1):
                assert iob_disc_labels[i] == "O"  # no overlapping annotations are expected
                if subtask == 'ner':
                    iob_disc_labels[i] = "I"
                elif subtask == 'norm':
                    iob_disc_labels[i] = "I" + "-" + row['code']

        ## Subsequent fragments
        ann_loc_len = len(ann_loc_split)
        for ann_i in range(1, ann_loc_len):
            loc_start = int(ann_loc_split[ann_i].split(' ')[0])
            loc_end = int(ann_loc_split[ann_i].split(' ')[1])
            # First subtoken/word of annotation
            tok_start = np.where(arr_start_end[:, 0] <= loc_start)[0][-1]  # last subtoken/word <= annotation start
            # Last subtoken/word of annotation
            tok_end = np.where(arr_start_end[:, 1] >= loc_end)[0][0]  # first subtoken/word >= annotation end
            assert tok_start <= tok_end
            # Annotate first subtoken/word
            assert iob_disc_labels[tok_start] == "O"  # no overlapping annotations are expected
            if subtask == 'ner':
                iob_disc_labels[tok_start] = "I"
            elif subtask == 'norm':
                iob_disc_labels[tok_start] = "I" + "-" + row['code']

            if tok_start < tok_end:
                # Annotation spanning multiple subtokens/words
                for i in range(tok_start + 1, tok_end + 1):
                    assert iob_disc_labels[i] == "O"  # no overlapping annotations are expected
                    if subtask == 'ner':
                        iob_disc_labels[i] = "I"
                    elif subtask == 'norm':
                        iob_disc_labels[i] = "I" + "-" + row['code']

    return [iob_disc_labels]


def convert_word_token(word_text, word_start_end, word_labels, tokenizer, ign_value, strategy, word_pos):
    """
    Given a list of words, the function converts them to a list of subtokens.
    Implemented strategies: word-all, word-first, word-first-x.
    """
    res_sub_token, res_start_end, res_word_id = [], [], []
    # Multiple labels
    res_labels = [[] for lab_i in range(len(word_labels))]
    for i in range(len(word_text)):
        w_text = word_text[i]
        w_start_end = word_start_end[i]
        sub_token, _ = start_end_tokenize(text=w_text, tokenizer=tokenizer, start_pos=w_start_end[0])
        tok_start_end = [w_start_end] * len(
            sub_token)  # using the word start-end pair as the start-end position of the subtokens
        tok_word_id = [i + word_pos] * len(sub_token)
        res_sub_token.extend(sub_token)
        res_start_end.extend(tok_start_end)
        res_word_id.extend(tok_word_id)
        # Multiple labels
        for lab_i in range(len(word_labels)):
            w_label = word_labels[lab_i][i]
            if strategy.split('-')[1] == "all":
                res_labels[lab_i].extend([w_label] * len(sub_token))
            else:
                subtk_value = "X" if strategy.split('-')[-1] == "x" else ign_value
                res_labels[lab_i].extend([w_label] + [subtk_value] * (len(sub_token) - 1))

    return res_sub_token, res_start_end, res_labels, res_word_id


def start_end_tokenize_ner(text, max_seq_len, tokenizer, start_pos, df_ann, ign_value, strategy="word-all", cased=True,
                           word_pos=0,
                           subtask='ner', code_strat='ign'):
    """
    Given an input text, it returns a list of lists containing the adjacent sequences of subtokens.
    return: list of lists, shape [n_sequences, n_subtokens] (out_sub_token, out_start_end, out_word_id)
            list of lists of lists, shape [n_outputs, n_sequences, n_subtokens] (out_labels)
    """

    out_sub_token, out_start_end, out_labels, out_word_id = [], [], [], []
    if strategy.split('-')[0] == "word":
        # Apply whitespace and punctuation pre-tokenization to extract the words from the input text
        word_text, word_chr_start_end = word_start_end(text=text, start_i=start_pos, cased=cased)
        assert len(word_text) == len(word_chr_start_end)
        if len(subtask.split('-')) == 1:
            # Obtain IOB-2/IOB-Code labels at word-level
            word_labels = ner_iob2_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, subtask=subtask)
        elif subtask.split('-')[1] == "iob_code":
            # Multiple labels
            word_labels = norm_iob2_code_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann,
                                                  ign_value=ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_disc":
            # subtask possible values: norm-ner-iob_disc, iob_disc
            word_labels = ner_iob2_disc_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann,
                                                 subtask=subtask.split('-')[0])

        for lab_i in range(len(word_labels)):
            assert len(word_labels[lab_i]) == len(word_text)

        # Convert word-level arrays to subtoken-level
        sub_token, start_end, labels, word_id = convert_word_token(word_text=word_text,
                                                                   word_start_end=word_chr_start_end,
                                                                   word_labels=word_labels, tokenizer=tokenizer,
                                                                   ign_value=ign_value, strategy=strategy,
                                                                   word_pos=word_pos)
    else:
        raise Exception('Strategy not implemented!')

    assert len(sub_token) == len(start_end) == len(word_id)
    # Multiple labels
    for lab_i in range(len(labels)):
        out_labels.append([])
        assert len(labels[lab_i]) == len(sub_token)

    # Re-split large sub-tokens sequences
    for i in range(0, len(sub_token), max_seq_len):
        out_sub_token.append(sub_token[i:i + max_seq_len])
        out_start_end.append(start_end[i:i + max_seq_len])
        out_word_id.append(word_id[i:i + max_seq_len])
        # Multiple labels
        for lab_i in range(len(labels)):
            out_labels[lab_i].append(labels[lab_i][i:i + max_seq_len])

    return out_sub_token, out_start_end, out_labels, out_word_id


def ss_sep_start_end_tokenize_ner(ss_start_end, max_seq_len, text, tokenizer, df_ann, ign_value, strategy="word-all",
                                  cased=True, subtask='ner', code_strat='ign'):
    """
    ss_start_end: list of tuples, where each tuple contains the start-end character positions pair of
                  the split sentences from the input document text.
    text: document text.

    return: 4 lists of lists, the first for the sub-tokens from the re-split sentences, the second for the
            start-end char positions pairs of the sub-tokens from the re-split sentences, the third for
            the IOB-2/IOB-Code labels associated to the sub-tokens from the re-split sentences, and the forth for the
            word id of each sub-token.
    """
    # Firstly, the whole text is tokenized (w/o re-splitting, see max_seq_len arg)
    txt_sub_token, txt_start_end, txt_labels, txt_word_id = start_end_tokenize_ner(text=text, max_seq_len=int(1e22),
                                                                                   tokenizer=tokenizer, start_pos=0,
                                                                                   df_ann=df_ann, ign_value=ign_value,
                                                                                   strategy=strategy, cased=cased,
                                                                                   word_pos=0, subtask=subtask,
                                                                                   code_strat=code_strat)
    assert len(txt_sub_token) == 1  # no re-splitting was performed
    txt_sub_token, txt_start_end, txt_word_id = txt_sub_token[0], txt_start_end[0], txt_word_id[0]
    for lab_i in range(len(txt_labels)):
        txt_labels[lab_i] = txt_labels[lab_i][0]
    # Then, the token sequences are split according to the SS information
    arr_txt_start_end = np.array(txt_start_end)
    ss_cur_sub_token, ss_cur_start_end, ss_cur_word_id = [], [], []
    ss_cur_labels = [[] for lab_i in range(len(txt_labels))]  # multiple labels
    start_tok = last_tok = 0
    for _, ss_end in ss_start_end:
        assert start_tok < len(txt_sub_token)  # never reach the end w/o finding all SS
        # We identify the position of the last subtoken of the current SS
        last_tok = np.where(arr_txt_start_end[start_tok:, 1] <= ss_end)[0][-1]
        last_tok += start_tok  # since in np.where operation start_tok is the initial position (index 0)
        # We add the current SS
        ss_cur_sub_token.append(txt_sub_token[start_tok:last_tok + 1])
        ss_cur_start_end.append(txt_start_end[start_tok:last_tok + 1])
        ss_cur_word_id.append(txt_word_id[start_tok:last_tok + 1])
        # Multiple labels
        for lab_i in range(len(ss_cur_labels)):
            ss_cur_labels[lab_i].append(txt_labels[lab_i][start_tok:last_tok + 1])
        start_tok = last_tok + 1

    # Re-split large sub-tokens sequences
    out_sub_token, out_start_end, out_word_id = [], [], []
    out_labels = [[] for lab_i in range(len(ss_cur_labels))]  # multiple labels
    for i in range(len(ss_cur_sub_token)):
        for j in range(0, len(ss_cur_sub_token[i]), max_seq_len):
            out_sub_token.append(ss_cur_sub_token[i][j:j + max_seq_len])
            out_start_end.append(ss_cur_start_end[i][j:j + max_seq_len])
            out_word_id.append(ss_cur_word_id[i][j:j + max_seq_len])
            # Multiple labels
            for lab_i in range(len(out_labels)):
                out_labels[lab_i].append(ss_cur_labels[lab_i][i][j:j + max_seq_len])

    return out_sub_token, out_start_end, out_labels, out_word_id


def ss_start_end_tokenize_ner(ss_start_end, max_seq_len, text, tokenizer, df_ann, ign_value, strategy="word-all",
                              cased=True, subtask='ner', code_strat='ign'):
    """
    ss_start_end: list of tuples, where each tuple contains the start-end character positions pair of
                  the split sentences from the input document text.
    text: document text.

    return: 4 lists of lists, the first for the sub-tokens from the re-split sentences, the second for the
            start-end char positions pairs of the sub-tokens from the re-split sentences, the third for
            the IOB-2/IOB-Code labels associated to the sub-tokens from the re-split sentences, and the forth for the
            word id of each sub-token.
    """
    out_sub_token, out_start_end, out_labels, out_word_id = [], [], [], []
    n_ss_words = 0
    for ss_start, ss_end in ss_start_end:
        ss_text = text[ss_start:ss_end]
        # annotations spanning multiple adjacent sentences are not considered
        ss_ann = df_ann[(df_ann['start'] >= ss_start) & (df_ann['end'] <= ss_end)]
        ss_sub_token, ss_start_end, ss_labels, ss_word_id = start_end_tokenize_ner(text=ss_text,
                                                                                   max_seq_len=max_seq_len,
                                                                                   tokenizer=tokenizer,
                                                                                   start_pos=ss_start, df_ann=ss_ann,
                                                                                   ign_value=ign_value,
                                                                                   strategy=strategy, cased=cased,
                                                                                   word_pos=n_ss_words, subtask=subtask,
                                                                                   code_strat=code_strat)
        out_sub_token.extend(ss_sub_token)
        out_start_end.extend(ss_start_end)
        out_word_id.extend(ss_word_id)
        if len(out_labels) == 0:  # first iteration (dirty, as the number of output tensors is not previously defined)
            out_labels = [[] for lab_i in range(len(ss_labels))]
        for lab_i in range(len(ss_labels)):
            out_labels[lab_i].extend(ss_labels[lab_i])

        # We update the number of words contained in the document so far
        n_ss_words = ss_word_id[-1][-1] + 1

    return out_sub_token, out_start_end, out_labels, out_word_id


def ss_fragment_greedy_ner(ss_token, ss_start_end, ss_labels, ss_word_id, max_seq_len):
    """
    Same as ss_fragment_greedy but also including a labels and word-id arrays
    """
    frag_token, frag_start_end, frag_word_id = [[]], [[]], [[]]
    # Multiple labels
    frag_labels = [[[]] for lab_i in range(len(ss_labels))]

    i = 0
    while i < len(ss_token):
        assert len(ss_token[i]) <= max_seq_len
        if len(frag_token[-1]) + len(ss_token[i]) > max_seq_len:
            # Fragment is full, so create a new empty fragment
            frag_token.append([])
            frag_start_end.append([])
            frag_word_id.append([])
            # Multiple labels
            for lab_i in range(len(ss_labels)):
                frag_labels[lab_i].append([])

        frag_token[-1].extend(ss_token[i])
        frag_start_end[-1].extend(ss_start_end[i])
        frag_word_id[-1].extend(ss_word_id[i])
        # Multiple labels
        for lab_i in range(len(ss_labels)):
            frag_labels[lab_i][-1].extend(ss_labels[lab_i][i])

        i += 1

    return frag_token, frag_start_end, frag_labels, frag_word_id


def format_token_ner(token_list, label_list, tokenizer, seq_len, lab_encoder_list, ign_value, fasttext_strat):
    """
    Given a list of sub-tokens and their assigned NER-labels, as well as a tokenizer, it returns their corresponding lists of
    indices, attention masks, tokens types and transformed labels. Padding is added as appropriate.
    """
    type_tokenizer = str(type(tokenizer))
    if 'transformers' in type_tokenizer:
        token_ids = tokenizer.convert_tokens_to_ids(token_list)
        # Add [CLS] and [SEP] tokens (single sequence)
        token_ids = tokenizer.build_inputs_with_special_tokens(token_ids)

        # Generate attention mask
        token_len = len(token_ids)
        attention_mask = [1] * token_len

        # Generate token types
        token_type = [0] * token_len

        # Add special tokens labels
        # Multiple labels
        token_labels = []
        for lab_i in range(len(label_list)):
            token_labels.append([ign_value] + [lab_encoder_list[lab_i][label] if label != ign_value else label \
                                               for label in label_list[lab_i]] + [ign_value])
            assert len(token_labels[lab_i]) == token_len

        # Padding
        pad_len = seq_len - token_len
        token_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        token_type += [0] * pad_len


    elif 'fasttext' in type_tokenizer:
        # Add special tokens labels
        # Multiple labels
        token_labels = []
        for lab_i in range(len(label_list)):
            token_labels.append([lab_encoder_list[lab_i][label] if label != ign_value else label \
                                 for label in label_list[lab_i]])

        # Implement differently according to the fine-tuning/freezed strategy
        if fasttext_strat == "ft":
            # Fine-tuning strategy (with zero-padding)
            word_id_offset = 2  # 0 = pad, -1 -> 1 = unk token, 0 -> 2 = first known token, etc.
            token_ids = [tokenizer.get_word_id(word) + word_id_offset for word in token_list]
            token_len = len(token_ids)
            attention_mask, token_type = [], []

            # Padding
            pad_len = seq_len - token_len
            token_ids += [0] * pad_len  # zero-padding

        elif fasttext_strat == "freeze":
            # Freezed embeddings strategy (with np.zeros padding)
            token_ids = [tokenizer.get_word_vector(word) for word in token_list]  # shape: (n_sub_tok_i, dim)
            token_len = len(token_ids)
            attention_mask, token_type = [], []

            # Padding
            pad_len = seq_len - token_len
            token_ids += [np.zeros(tokenizer.get_dimension())] * pad_len  # zero-padding, final shape: (seq_len, dim)

    # Multiple labels
    for lab_i in range(len(label_list)):
        token_labels[lab_i].extend([ign_value] * pad_len)

    return token_ids, attention_mask, token_type, token_labels


from copy import deepcopy


def ss_create_input_data_ner(df_text, text_col, df_ann, df_ann_text, doc_list, ss_dict, tokenizer, lab_encoder_list,
                             text_label_encoder, seq_len, ign_value,
                             strategy="word-all", greedy=False, cased=True, subtask='ner', code_strat='ign',
                             fasttext_strat="ft", ss_sep=False):
    """
    This function generates the data needed to fine-tune a transformer model on a multi-class token classification task,
    such as Cantemist-NER subtask, following the IOB-2 annotation format.

    ss_sep input arg: if True, firstly, the tokenization of the whole document is performed, and then,
                      the token sequences are split according to the SS information.
                      (Modification added on 26/09/22, for de-identification project)
    """

    indices, attention_mask, token_type, labels, text_labels, n_fragments, start_end_offsets, word_ids = [], [], [], [], [], [], [], []
    sub_tok_max_seq_len = seq_len
    type_tokenizer = str(type(tokenizer))
    if 'transformers' in type_tokenizer: sub_tok_max_seq_len -= 2
    for doc in doc_list:
        # Extract doc annotation
        doc_ann = df_ann[df_ann["doc_id"] == doc]
        # Text classification
        doc_ann_text = df_ann_text[df_ann_text["doc_id"] == doc]
        # Extract doc text
        doc_text = df_text[df_text["doc_id"] == doc][text_col].values[0]
        ## Generate annotated subtokens sequences
        if ss_dict is not None:
            # Perform sentence split (SS) on doc text
            doc_ss = ss_dict[doc]  # SS start-end pairs of the doc text
            if ss_sep:
                doc_ss_token, doc_ss_start_end, doc_ss_label, doc_ss_word_id = ss_sep_start_end_tokenize_ner(
                    ss_start_end=doc_ss,
                    max_seq_len=sub_tok_max_seq_len, text=doc_text,
                    tokenizer=tokenizer, df_ann=doc_ann, ign_value=ign_value, strategy=strategy, cased=cased,
                    subtask=subtask, code_strat=code_strat)
            else:
                doc_ss_token, doc_ss_start_end, doc_ss_label, doc_ss_word_id = ss_start_end_tokenize_ner(
                    ss_start_end=doc_ss,
                    max_seq_len=sub_tok_max_seq_len, text=doc_text,
                    tokenizer=tokenizer, df_ann=doc_ann, ign_value=ign_value, strategy=strategy, cased=cased,
                    subtask=subtask, code_strat=code_strat)
            assert len(doc_ss_token) == len(doc_ss_start_end) == len(doc_ss_word_id)
            # Multiple labels
            for lab_i in range(len(doc_ss_label)):
                assert len(doc_ss_label[lab_i]) == len(doc_ss_token)

            if greedy:
                # Split the list of sub-tokens sentences into sequences comprising multiple sentences
                frag_token, frag_start_end, frag_label, frag_word_id = ss_fragment_greedy_ner(ss_token=doc_ss_token,
                                                                                              ss_start_end=doc_ss_start_end,
                                                                                              ss_labels=doc_ss_label,
                                                                                              ss_word_id=doc_ss_word_id,
                                                                                              max_seq_len=sub_tok_max_seq_len)
            else:
                frag_token = deepcopy(doc_ss_token)
                frag_start_end = deepcopy(doc_ss_start_end)
                frag_label = deepcopy(doc_ss_label)
                frag_word_id = deepcopy(doc_ss_word_id)
        else:
            # Generate annotated sequences using text-stream strategy (without considering SS)
            frag_token, frag_start_end, frag_label, frag_word_id = start_end_tokenize_ner(text=doc_text,
                                                                                          max_seq_len=sub_tok_max_seq_len,
                                                                                          tokenizer=tokenizer,
                                                                                          start_pos=0, df_ann=doc_ann,
                                                                                          ign_value=ign_value,
                                                                                          strategy=strategy,
                                                                                          cased=cased, word_pos=0,
                                                                                          subtask=subtask,
                                                                                          code_strat=code_strat)

        assert len(frag_token) == len(frag_start_end) == len(frag_word_id)
        # Multiple labels
        for lab_i in range(len(frag_label)):
            assert len(frag_label[lab_i]) == len(frag_token)
        # Store the start-end char positions of all the sequences
        start_end_offsets.extend(frag_start_end)
        # Store the sub-tokens word ids of all the sequences
        word_ids.extend(frag_word_id)
        # Store the number of sequences of each doc text
        n_fragments.append(len(frag_token))
        ## Subtokens sequences formatting
        # Multiple labels
        if len(labels) == 0:
            labels = [[] for lab_i in range(
                len(frag_label))]  # first iteration (dirty, as the number of output tensors is not previously defined)
        for seq_i in range(len(frag_token)):
            f_token = frag_token[seq_i]
            f_start_end = frag_start_end[seq_i]
            f_word_id = frag_word_id[seq_i]
            # Multiple labels
            f_label = []
            for lab_i in range(len(frag_label)):
                f_label.append(frag_label[lab_i][seq_i])
            # sequence length is assumed to be <= SEQ_LEN-2
            assert len(f_token) == len(f_start_end) == len(f_word_id) <= sub_tok_max_seq_len
            # Multiple labels
            for lab_i in range(len(f_label)):
                assert len(f_label[lab_i]) == len(f_token)
            f_id, f_att, f_type, f_label = format_token_ner(token_list=f_token, label_list=f_label,
                                                            tokenizer=tokenizer, seq_len=seq_len,
                                                            lab_encoder_list=lab_encoder_list, ign_value=ign_value,
                                                            fasttext_strat=fasttext_strat)

            # Text classification
            text_frag_labels = []
            # start-end char positions of the whole fragment, i.e. the start position of the first
            # sub-token and the end position of the last sub-token
            frag_start, frag_end = f_start_end[0][0], f_start_end[-1][1]
            for j in range(doc_ann_text.shape[0]):
                doc_ann_cur = doc_ann_text.iloc[j]  # current annotation
                # Add the annotations whose text references are contained within the fragment
                if doc_ann_cur['start'] < frag_end and doc_ann_cur['end'] > frag_start:
                    text_frag_labels.append(doc_ann_cur['code'])
            text_labels.append(text_frag_labels)

            indices.append(f_id)
            attention_mask.append(f_att)
            token_type.append(f_type)
            # Multiple labels
            for lab_i in range(len(f_label)):
                labels[lab_i].append(f_label[lab_i])

    return np.array(indices), np.array(attention_mask), np.array(token_type), np.array(labels), \
        text_label_encoder.transform(text_labels), np.array(n_fragments), start_end_offsets, word_ids


## NER performance evaluation

from sklearn.preprocessing import normalize


def word_seq_preds(tok_seq_word_id, tok_seq_preds, tok_seq_start_end, strategy):
    """
    Implemented strategies: "word-first", "word-max", "word-prod", "word-sum", "word-all-crf", "word-first-crf",
                            "word-first-x-crf".
    """

    # Convert subtoken-level predictions to word-level predictions
    arr_word_seq_start_end = []
    # Multiple labels
    arr_word_seq_preds = [[] for lab_i in range(len(tok_seq_preds))]
    left = 0
    while left < len(tok_seq_word_id):
        cur_word_id = tok_seq_word_id[left]
        right = left + 1
        while right < len(tok_seq_word_id):
            if tok_seq_word_id[right] != cur_word_id:
                break
            right += 1
        # cur_word_id spans from left to right - 1 subtoken positions
        assert len(set(tok_seq_start_end[
                       left:right])) == 1  # start-end pos of the subtokens correspond to the word start-end pos
        arr_word_seq_start_end.append(tok_seq_start_end[left])

        # Multiple labels
        for lab_i in range(len(tok_seq_preds)):
            if strategy.split('-')[-1] == "max":
                # max of predictions made in all subtokens of the word
                arr_word_seq_preds[lab_i].append(np.max(tok_seq_preds[lab_i][left:right], axis=0))

            elif strategy.split('-')[-1] == "prod":
                # product of predictions made in all subtokens of the word
                arr_word_seq_preds[lab_i].append(np.prod(tok_seq_preds[lab_i][left:right], axis=0))

            elif strategy.split('-')[-1] == "sum":
                # sum of predictions made in all subtokens of the word
                arr_word_seq_preds[lab_i].append(np.sum(tok_seq_preds[lab_i][left:right], axis=0))

            elif strategy.split('-')[-1] == "sum_norm":
                # sum of predictions made in all subtokens of the word
                arr_word_seq_preds[lab_i].append(normalize(
                    np.sum(tok_seq_preds[lab_i][left:right], axis=0).reshape(1, -1),
                    norm='l1',
                    axis=1
                )[0])

            elif '-'.join(strategy.split('-')[-2:]) == "all-crf":
                # label obtaining the relative majority from the predictions made in all subtokens of the word
                # (labels are assumed to be int)
                arr_word_seq_preds[lab_i].append(np.argmax(np.bincount(tok_seq_preds[lab_i][left:right])))

            elif '-'.join(strategy.split('-')[-2:]) == "first-crf":
                # label predicted on the first subtoken of the word
                arr_word_seq_preds[lab_i].append(
                    tok_seq_preds[lab_i][cur_word_id])  # CRF only predicts the first subtoken of each word
            elif strategy.split('-')[1] == "first":  # word-first, word-first-x-crf
                # predictions made on the first subtoken of the word
                arr_word_seq_preds[lab_i].append(tok_seq_preds[lab_i][left])

            else:
                raise Exception('Word strategy not implemented!')

        left = right

    assert cur_word_id == tok_seq_word_id[-1]

    return arr_word_seq_preds, arr_word_seq_start_end


def ner_iob2_extract_seq_preds(doc_id, seq_preds, seq_start_end, df_text, text_col, subtask='ner',
                               strategy='word-first'):
    """
    seq_preds: it is assumed to be a list containing a single list (single label, either IOB or IOB-Code),
    e.g. NER: [[("B"), ("O")]]; NORM: [[("B-8000/3", 0.87), ("O", 0.6)]] (non-CRF), [[("B-8000/3"), ("O")]] (CRF)

    subtask: ner, norm, norm-mention
    """

    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0].split('-')[0] == "B":
            right = left + 1
            while right < len(seq_preds[0]):
                if seq_preds[0][right][0].split('-')[0] != "I":
                    break
                right += 1
            # Add NER annotation
            res.append({'clinical_case': doc_id, 'start': seq_start_end[left][0], 'end': seq_start_end[right - 1][1],
                        'text': df_text[df_text['doc_id'] == doc_id][text_col].values[0]})

            if subtask.split('-')[0] == 'norm':
                if subtask == 'norm':
                    if strategy.split('-')[-1] != "crf":
                        # Extract probabilities of the labels predicted within the annotation (from left to right - 1 pos)
                        ann_lab_prob = np.array([pair[1] for pair in seq_preds[0][left:right]])
                        # Select the label with the maximum probability
                        max_lab = left + np.argmax(ann_lab_prob)
                        code_pred = seq_preds[0][max_lab][0].split('-')[1]
                    else:
                        # Extract codes predicted in the annotation (from left to right - 1 pos)
                        ann_codes = [pred[0].split('-')[1] for pred in seq_preds[0][left:right]]
                        # Select the most frequently predicted code within the annotation
                        codes_uniq, codes_freq = np.unique(ann_codes, return_counts=True)
                        code_pred = codes_uniq[np.argmax(codes_freq)]

                elif subtask == 'norm-mention':
                    code_pred = seq_preds[0][left][0].split('-')[1]

                # Add NORM annotation
                res[-1]['code_pred'] = code_pred

            left = right  # next sub-token different from "I", or len(seq_preds[0]) (out of bounds)
        else:
            left += 1

    return res


def mention_seq_preds(seq_preds, mention_strat='max'):
    """
    Given the word-level coding-predictions (labels probabilities) made in a detected mention, the function returns a
    single coding-prediction (labels probabilities) made for the whole mention.

    seq_preds: shape n_words (in the mention) x n_labels (1 for CRF)

    mention_strat: first, max, prod, all-crf.
    """

    res = None
    if mention_strat == "first":
        # Select the coding-prediction made for the first word of the mention
        res = seq_preds[0]
    elif mention_strat == "max":
        res = np.max(seq_preds, axis=0)
    elif mention_strat == "prod":
        res = np.prod(seq_preds, axis=0)
    elif mention_strat == "sum":
        res = np.sum(seq_preds, axis=0)
    elif mention_strat == "all-crf":
        # Select the most frequently predicted coding-label within the mention
        res = np.argmax(np.bincount(seq_preds))  # predicted labels are assumed to be int
    else:
        raise Exception('Mention strategy not implemented!')

    return res


def norm_iob2_code_extract_seq_preds(doc_id, seq_preds, seq_start_end, df_text, text_col, code_lab_decoder_list,
                                     strategy='word-first', subtask='norm-iob_code', mention_strat='max'):
    """
    seq_preds: it is assumed to be a list containing two lists (double label, IOB + Code), e.g. [[("B"), ("O")],
    [('8000/3', 0.87), ('8756/3H', 0.2)]] (non-CRF), [[("B"), ("O")], [('8000/3'), ('8756/3H')]] (CRF)

    subtask: norm-iob_code, norm_iob_code-mention

    code_lab_decoder_list: [code_lab_decoder]
    """

    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "B":
            right = left + 1
            while right < len(seq_preds[0]):
                if seq_preds[0][right][0] != "I":
                    break
                right += 1
            # Add NER annotation
            res.append({'clinical_case': doc_id, 'start': seq_start_end[left][0], 'end': seq_start_end[right - 1][1],
                        'text': df_text[df_text['doc_id'] == doc_id][text_col].values[0]})
            # Coding-predictions made on the detected mention (from left to right - 1 pos)
            mention_code_preds = seq_preds[1][left:right]
            if '-'.join(subtask.split('-')[1:]) == 'iob_code':
                if strategy.split('-')[-1] != "crf":
                    code_pred = code_lab_decoder_list[0][np.argmax(mention_seq_preds(seq_preds=mention_code_preds,
                                                                                     mention_strat=mention_strat))]

                else:
                    code_pred = code_lab_decoder_list[0][
                        mention_seq_preds(seq_preds=mention_code_preds, mention_strat='all-crf')]

            elif '-'.join(subtask.split('-')[1:]) == 'iob_code-mention':
                label_pred = mention_seq_preds(seq_preds=mention_code_preds, mention_strat='first')
                if strategy.split('-')[-1] != "crf":
                    label_pred = np.argmax(label_pred)
                code_pred = code_lab_decoder_list[0][label_pred]

            # Add NORM annotation
            res[-1]['code_pred'] = code_pred

            left = right  # next sub-token different from "I", or len(seq_preds[0]) (out of bounds)
        else:
            left += 1

    return res


def ner_iob2_disc_extract_seq_preds(doc_id, seq_preds, seq_start_end, df_text, text_col, subtask='ner',
                                    strategy='word-all'):
    """
    seq_preds: it is assumed to be a list containing 1 list (IOB (disc))
    """

    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0].split('-')[0] == "B":
            ## First fragment
            right = left + 1
            while (right < len(seq_preds[0])) and (seq_preds[0][right][0].split('-')[0] == "I"):
                right += 1

            # Save pos
            ann_pos = str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right - 1][1])
            ann_iter_pos = list(range(left, right))

            left = right
            while (left < len(seq_preds[0])) and (seq_preds[0][left][0].split('-')[0] != "B"):
                if seq_preds[0][left][0].split('-')[0] == "I":
                    ## Subsequent fragment
                    right = left + 1
                    while (right < len(seq_preds[0])) and (seq_preds[0][right][0].split('-')[0] == "I"):
                        right += 1

                    # Save pos
                    ann_pos += ';' + str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right - 1][1])
                    ann_iter_pos += list(range(left, right))

                    left = right

                else:
                    left += 1

            # Add NER annotation
            # For discontinuous annotations, just keep the first and last offset
            ann_start_pos = ann_pos.split(' ')[0]
            ann_end_pos = ann_pos.split(' ')[-1]
            res.append({'clinical_case': doc_id, 'start': int(ann_start_pos), 'end': int(ann_end_pos),
                        'text': df_text[df_text['doc_id'] == doc_id][text_col].values[0],
                        'location': ann_pos})

            if subtask.split('-')[0] == 'norm':
                if strategy.split('-')[-1] != "crf":
                    # Extract probabilities of the labels predicted within the annotation
                    ann_lab_prob = np.array([seq_preds[0][pos][1] for pos in ann_iter_pos])
                    # Select the position of the label with the maximum probability
                    max_lab = ann_iter_pos[np.argmax(ann_lab_prob)]
                    code_pred = seq_preds[0][max_lab][0].split('-')[1]
                else:
                    # Extract codes predicted within the annotation
                    ann_codes = [seq_preds[0][pos][0].split('-')[1] for pos in ann_iter_pos]
                    # Select the most frequently predicted code within the annotation
                    codes_uniq, codes_freq = np.unique(ann_codes, return_counts=True)
                    code_pred = codes_uniq[np.argmax(codes_freq)]

                # Add NORM annotation
                res[-1]['code_pred'] = code_pred

        else:
            left += 1

    return res


def seq_ner_preds_brat_format(doc_list, fragments, arr_start_end, arr_word_id, arr_preds, strategy="word-first",
                              crf_mask_seq_len=None,
                              type_tokenizer='transformers'):
    """
    Implemented strategies: "word-first", "word-max", "word-prod", "word-all-crf", "word-first-crf", "word-first-x-crf".
    """

    arr_doc_seq_start_end = []
    # Multiple labels
    arr_doc_seq_preds = [[] for lab_i in range(len(arr_preds))]
    i = 0
    for d in range(len(doc_list)):
        n_frag = fragments[d]
        # Extract subtoken-level arrays for each document (joining adjacent fragments)
        doc_tok_start_end = [ss_pair for frag in arr_start_end[i:i + n_frag] for ss_pair in frag]
        doc_tok_word_id = [w_id for frag in arr_word_id[i:i + n_frag] for w_id in frag]
        assert len(doc_tok_start_end) == len(doc_tok_word_id)

        if strategy.split('-')[0] == "word":
            # Extract subtoken-level predictions, ignoring special tokens (CLS, SEQ, PAD)
            # (CLS, SEP only for transformers, not for fasttext)
            # Multiple labels
            doc_tok_preds = []
            inf = sup = 1
            if type_tokenizer == 'fasttext':
                inf = sup = 0
            for lab_i in range(len(arr_preds)):
                if strategy.split('-')[-1] != "crf":
                    # doc_tok_preds[lab_i] shape: n_tok (per doc) x n_labels (3 for NER, 2*n_codes + 1 for NER-Norm)
                    doc_tok_preds.append(np.array([preds for j in range(i, i + n_frag) \
                                                   for preds in arr_preds[lab_i][j][inf:len(arr_start_end[j]) + sup]]))
                    assert doc_tok_preds[-1].shape[0] == len(doc_tok_start_end)
                else:
                    # arr_preds does not contain predictions made on "ignored" tokens
                    # (either special tokens or secondary tokens for "word-first" strategy),
                    # but it contains predictions for right-padding-CRF tokens.
                    # crf_mask_seq_len is expected not to be None, indicating the
                    # number of "not right-padded" tokens in each fragment;
                    # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
                    # for all outputs (this may be changed when implementing "mention-first" approach).
                    # doc_tok_preds shape: n_tok (per doc) x 1 (int label)
                    doc_tok_preds.append(np.array([preds for j in range(i, i + n_frag) \
                                                   for preds in arr_preds[lab_i][j][:crf_mask_seq_len[j]]]))
                    if strategy.split('-')[-2] == "all":
                        assert doc_tok_preds[-1].shape[0] == len(doc_tok_start_end)
                    elif strategy.split('-')[-2] == "first":
                        assert doc_tok_preds[-1].shape[0] == (doc_tok_word_id[-1] + 1)
                    elif '-'.join(strategy.split('-')[-3:-1]) == "first-x":
                        assert doc_tok_preds[-1].shape[0] == len(doc_tok_start_end)
                    else:
                        raise Exception('Strategy not implemented!')

            # Convert subtoken-level arrays to word-level
            doc_word_seq_preds, doc_word_seq_start_end = word_seq_preds(tok_seq_word_id=doc_tok_word_id,
                                                                        tok_seq_preds=doc_tok_preds,
                                                                        tok_seq_start_end=doc_tok_start_end,
                                                                        strategy=strategy)
            assert len(doc_word_seq_start_end) == (doc_tok_word_id[-1] + 1)

            # Multiple labels
            for lab_i in range(len(arr_preds)):
                assert len(doc_word_seq_preds[lab_i]) == len(doc_word_seq_start_end)
                arr_doc_seq_preds[lab_i].append(doc_word_seq_preds[
                                                    lab_i])  # final shape: n_doc x n_words (per doc) x [n_labels (e.g. 3 for NER) or 1 (CRF)]
            arr_doc_seq_start_end.append(
                doc_word_seq_start_end)  # final shape: n_doc x n_words (per doc) x 2 (start-end pair)

        else:
            raise Exception('Strategy not implemented!')

        i += n_frag

    return arr_doc_seq_preds, arr_doc_seq_start_end


def extract_seq_preds(arr_doc_seq_preds, arr_doc_seq_start_end, doc_list, lab_decoder_list, df_text, text_col,
                      subtask='ner', strategy="word-all", mention_strat='max', code_sep='/',
                      codes_pre_suf_mask=None, codes_pre_o_mask=None):
    """
    Implemented strategies: "word-first", "word-max", "word-prod", "word-all-crf", "word-first-crf".
    """

    n_output = len(arr_doc_seq_preds)
    ann_res = []
    for d in range(len(doc_list)):
        doc = doc_list[d]
        # Multiple labels
        doc_seq_preds = []
        for lab_i in range(n_output):
            doc_seq_preds.append(
                arr_doc_seq_preds[lab_i][d])  # shape: n_words (per doc) x [n_labels (e.g. 3 for NER) or 1 (CRF)]
        doc_seq_start_end = arr_doc_seq_start_end[d]  # shape: n_words (per doc) x 2 (start-end pair)
        doc_seq_lab = []
        if len(subtask.split('-')) == 1 or subtask.split('-')[1] in (
        'mention', 'iob_disc'):  # ner, norm, norm-mention, ner-iob_disc, norm-iob_disc
            # Single label (custom)
            if subtask.split('-')[0] == "ner":
                # IOB label
                if strategy.split('-')[-1] != "crf":
                    doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
                else:
                    doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])

            elif subtask.split('-')[0] == "norm":
                # IOB-Code label
                doc_seq_lab.append([])
                for i in range(len(doc_seq_preds[0])):
                    if strategy.split('-')[-1] != "crf":
                        max_j = np.argmax(doc_seq_preds[0][i])
                        # append both the predicted label and its probability
                        doc_seq_lab[0].append((lab_decoder_list[0][max_j],
                                               doc_seq_preds[0][i][max_j]))  # final shape: n_words (per doc) x 2
                    else:
                        doc_seq_lab[0].append(
                            (lab_decoder_list[0][doc_seq_preds[0][i]],))  # final shape: n_words (per doc) x 1

            else:
                raise Exception("Subtask not implemented!")

            if len(subtask.split('-')) == 1 or subtask.split('-')[1] != "iob_disc":
                ann_res.extend(
                    ner_iob2_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end,
                                               df_text=df_text, text_col=text_col, subtask=subtask, strategy=strategy))

            else:
                ann_res.extend(
                    ner_iob2_disc_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end,
                                                    df_text=df_text, text_col=text_col, subtask=subtask.split('-')[0],
                                                    strategy=strategy))


        elif subtask.split('-')[1] == "iob_code":
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            # Code predictions
            # Mention strategy
            doc_seq_lab.append(doc_seq_preds[1])  # final shape: n_words (per doc) x n_labels (1 for CRF)
            ann_res.extend(
                norm_iob2_code_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end,
                                                 df_text=df_text, text_col=text_col,
                                                 code_lab_decoder_list=lab_decoder_list[1:],
                                                 strategy=strategy, subtask=subtask, mention_strat=mention_strat))

        else:
            raise Exception("Subtask not implemented!")

    return ann_res


def ner_preds_brat_format(doc_list, fragments, preds, start_end, word_id, lab_decoder_list, df_text, text_col,
                          strategy="word-all", subtask="ner", crf_mask_seq_len=None, mention_strat='max',
                          type_tokenizer='transformers', code_sep='/',
                          codes_pre_suf_mask=None, codes_pre_o_mask=None):
    """
    Implemented strategies: "word-first", "word-max", "word-prod", "word-all-crf", "word-first-crf".
    """
    # Post-process the subtoken predictions for each document, obtaining word-level predictions
    arr_doc_seq_preds, arr_doc_seq_start_end = seq_ner_preds_brat_format(doc_list=doc_list, fragments=fragments,
                                                                         arr_start_end=start_end, arr_word_id=word_id,
                                                                         arr_preds=preds, strategy=strategy,
                                                                         crf_mask_seq_len=crf_mask_seq_len,
                                                                         type_tokenizer=type_tokenizer)

    # Extract the predicted mentions from the word-level predictions
    ann_res = extract_seq_preds(arr_doc_seq_preds=arr_doc_seq_preds, arr_doc_seq_start_end=arr_doc_seq_start_end,
                                doc_list=doc_list, lab_decoder_list=lab_decoder_list, df_text=df_text,
                                text_col=text_col, subtask=subtask, strategy=strategy, mention_strat=mention_strat,
                                code_sep=code_sep,
                                codes_pre_suf_mask=codes_pre_suf_mask, codes_pre_o_mask=codes_pre_o_mask)

    return pd.DataFrame(ann_res)


import shutil


def write_anon_ann(df_pred_ann, out_path):
    """
    Write a set of Anonimization-annotations from different documents in BRAT format.
    """

    # Create a new output directory
    if os.path.exists(out_path):
        shutil.rmtree(out_path, ignore_errors=True)
    os.mkdir(out_path)

    for doc in sorted(set(df_pred_ann['clinical_case'])):
        doc_pred_ann = df_pred_ann[df_pred_ann['clinical_case'] == doc]
        with open(out_path + doc + ".ann", "w") as out_file:
            i = 1
            for index, row in doc_pred_ann.iterrows():
                out_file.write("T" + str(i) + "\t" + row['code_pred'] + " " + row['location'] +
                               "\t" + row['text_ref'].replace("\n", " ") + "\n")
                i += 1
    return


import warnings


# Galén de-identification
def calculate_anon_single_metrics(gs, pred, span_col='location', subtask='norm'):
    """
    Micro-averaged classification metrics

    gs: df with columns "clinical_case, start, end, code_gs"
    pred: df with columns "clinical_case, start, end, code_pred"
    """
    pred_class = pred.copy()
    gs_class = gs.copy()

    if subtask == 'ner':
        pred_class["code_pred"] = "X"
        gs_class["code_gs"] = "X"

    if span_col not in pred_class.columns:
        pred_class[span_col] = pred_class.apply(lambda x: str(x['start']) + ' ' + str(x['end']), axis=1)
    if span_col not in gs_class.columns:
        gs_class[span_col] = gs_class.apply(lambda x: str(x['start']) + ' ' + str(x['end']), axis=1)

    # Predicted Positives:
    Pred_Pos = pred_class.drop_duplicates(subset=["clinical_case", span_col]).shape[0]

    # Gold Standard Positives:
    GS_Pos = gs_class.drop_duplicates(subset=["clinical_case", span_col]).shape[0]

    # Eliminate predictions not in GS (prediction needs to be in same clinical
    # case and to have the exact same offset to be considered valid!!!!)
    df_sel = pd.merge(pred_class, gs_class,
                      how="right",
                      on=["clinical_case", span_col])

    # Check if codes are equal
    df_sel["is_valid"] = \
        df_sel.apply(lambda x: (x["code_gs"] == x["code_pred"]), axis=1)

    # True Positives:
    TP = df_sel[df_sel["is_valid"] == True].shape[0]

    # Calculate Final Metrics:
    P = TP / Pred_Pos if Pred_Pos > 0 else 0
    R = TP / GS_Pos
    if (P + R) == 0:
        F1 = 0
    else:
        F1 = (2 * P * R) / (P + R)

    if (any([F1, P, R]) > 1):
        warnings.warn('Metric greater than 1! You have encountered an undetected bug!')

    return P, R, F1


def calculate_anon_metrics(gs, pred):
    """
    gs: df with columns "clinical_case, start, end, code_gs"
    pred: df with columns "clinical_case, start, end, code_pred"
    """

    res_dict = {}
    for code_class in sorted(set(gs["code_gs"])):
        gs_class = gs[gs["code_gs"] == code_class].copy()
        pred_class = pred[pred["code_pred"] == code_class].copy()

        P, R, F1 = calculate_anon_single_metrics(gs_class, pred_class)

        # Save results
        res_dict[code_class] = {'P': P, 'R': R, 'F1': F1}

    res_df = pd.DataFrame(res_dict).transpose()

    # Also return macro-averaged metrics
    return res_df, np.mean(res_df['P']), np.mean(res_df['R']), np.mean(res_df['F1'])


## NER loss

import tensorflow as tf


class TokenClassificationLoss(tf.keras.losses.Loss):
    """
    Code adapted from https://huggingface.co/transformers/_modules/transformers/modeling_tf_utils.html#TFTokenClassificationLoss
    """

    def __init__(self, from_logits=True, ignore_val=-100,
                 reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        self.from_logits = from_logits
        self.ignore_val = ignore_val
        self.reduction = reduction
        super(TokenClassificationLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred, sample_weight=None):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=self.from_logits, reduction=self.reduction
        )
        # make sure only labels that are not equal to self.ignore_val
        # are taken into account as loss
        active_loss = tf.reshape(y_true, (-1,)) != self.ignore_val
        reduced_preds = tf.boolean_mask(tf.reshape(y_pred, (-1, y_pred.shape[2])), active_loss)
        labels = tf.boolean_mask(tf.reshape(y_true, (-1,)), active_loss)

        return loss_fn(labels, reduced_preds, sample_weight=sample_weight)


## Tokenize documents.

from galennlp_tokens import Tokenizer
from galennlp_tools.flags import WordFlags
from galennlp_fragments.fragments import IndexedLine
from galennlp_fasttext.readonly import ReadonlyFasttext
from galennlp_corpus.paragraph.blocks import BlockJoiner
from galennlp_keras_utils.sequences.multilabel import MultiLabelTrainSequence


def extract_features(tokenizer, transformer, embedding, tok_line, text, doc_ann,
                     lab_encoder_list, num_class, vect_size, idx_line):
    """explore lines in document and write tokens"""
    # Numpy array with all line tokenize
    tok_vector = np.empty(shape=(0, seq_len))

    # TOKEN TRANSFORM
    tok_line = list(transformer.transform(tokenizer.parse(tok_line)))

    for (old_tok, new_tok) in tok_line:
        # Aux Tok_vector
        tok_vector_aux = np.empty(shape=(1, vect_size))
        tok_start = new_tok.fragments[0].document_chunk.start
        tok_end = tok_start + new_tok.fragments[0].document_chunk.size

        lab_true = 'O'

        # Get lab from the ann if it exists.
        for index, row in doc_ann.iterrows():
            if index not in set_lab:
                if len(new_tok.fragments) == 2:
                    tok_start = new_tok.fragments[1].document_chunk.start
                    tok_end = tok_start + new_tok.fragments[1].document_chunk.size

                if row['start'] <= tok_start < tok_end <= row['end']:
                    lab_true = row['code']
                    break

        # Get entity class.
        aux_classes = [0] * num_class
        aux_classes[int(lab_encoder_list[lab_true])] = 1

        # OUTPUT
        lim_inf = 0
        lim_sup = 1 + num_class
        tok_vector_aux[0, lim_inf:lim_sup] = np.array([idx_line] + aux_classes)

        # WORD FLAGS
        lim_inf = lim_sup
        lim_sup = lim_sup + 4
        tok_text = text[old_tok.document_roi().as_slice()]
        tok_vector_aux[0, lim_inf:lim_sup] = \
            WordFlags.to_one_hot(WordFlags.process(tok_text).to_bytes(1, "little"))[0][0]

        # TAGS FOR TOKENS
        lim_inf = lim_sup
        lim_sup = lim_sup + 12
        tok_vector_aux[0, lim_inf:lim_sup] = tag_format.to_one_hot(old_tok.token.tag)[0]

        # EMBEDDINGS
        lim_inf = lim_sup
        tok_vector_aux[0, lim_inf:] = embedding[new_tok.token.value]

        # UPDATE
        tok_vector = np.concatenate((tok_vector, tok_vector_aux), axis=0)

    return tok_vector


def tokenize_doc(tokenizer, transformer, embedding, joiner, docs_tokenizers,
                 doc_text, doc_ann, lab_encoder_list, num_class, vect_size, idx_line):
    # Iterate over blocks
    for line in index_lines(doc_text, joiner):
        # List all tokens in line
        tokens = []

        # Update with tokens in line
        tok_line = list(tokenizer.parse(line))
        tokens.append(tok_line)

        # Get tokens numpy vector
        tok_vector = extract_features(tokenizer, transformer, embedding, tok_line, line, doc_ann,
                                      lab_encoder_list, num_class, vect_size, idx_line)
        idx_line += 1

        docs_tokenizers = np.concatenate((docs_tokenizers, tok_vector), axis=0)

    return docs_tokenizers, idx_line


def index_lines(text: str, joiner) -> Iterator[IndexedLine]:
    lines = text.splitlines(keepends=True)
    return joiner.read_document(lines)


def process_data(df_text, df_ann, doc_list, lab_encoder_list, seq_len):
    # Tokenizer, transformer, word embedding and block joiner
    tokenizer = Tokenizer()
    transformer = TokenTransformer([], {})
    embedding = ReadonlyFasttext(str(fasttext), str(wordvectors), str(ngrams))
    joiner = BlockJoiner()

    # Vector size
    num_class = len(lab_encoder_list.keys())
    tok_size = 4 + 12 + seq_len
    # idx + class + flags + tags + embedding
    vect_size = 1 + num_class + tok_size

    # Final data
    docs_tokenizers = np.empty(shape=(0, vect_size))
    idx_line = 0

    for doc in doc_list:
        # Text classification
        doc_ann = df_ann[df_ann["doc_id"] == doc]
        # Extract doc text
        doc_text = df_text[df_text["doc_id"] == doc]["raw_text"].values[0]

        docs_tokenizers, idx_line = tokenize_doc(tokenizer, transformer, embedding, joiner, docs_tokenizers,
                                                 doc_text, doc_ann, lab_encoder_list, num_class, vect_size, idx_line)

    return MultiLabelTrainSequence(docs_tokenizers)


## Train the Recurrent Neural Network

from tensorflow.keras import models as k_models


def train_rnn_model(model, gen_train, gen_valid, batch_size, epochs, patience, null_class, threshold, model_path):
    best_idx = 0
    best_ref_value = -1
    for idx in range(epochs):
        history = model.fit(
            gen_train,
            batch_size=batch_size,
            epochs=1,
            validation_data=gen_valid,
            verbose=1
        )

        # Evaluate the model with validation set
        values, prediction = evaluate_model(model, gen_valid, null_class, threshold)
        ref_value = values["f1_micro"]
        if ref_value > best_ref_value:
            best_ref_value = ref_value
            model.save(model_path + '/best_model')

        if idx and (patience is not None) and ((idx - best_idx) > patience):
            break

    return k_models.load_model(model_path + '/best_model')


## Evaluate model

from sklearn import metrics


def evaluate_model(model, data_gen, null_class, threshold=0.5):
    """evaluar la red con el generador de test proporcionado

    model - keras model
    data_gen - dataset iterator
    null_class - índice de la clase nula (o None)
    """
    prediction = data_gen.predict(model)
    p_prob = prediction[:, 0, :]
    p_pred = (p_prob > threshold) + 0.0
    p_true = prediction[:, 1, :]
    valid_labels = list(range(p_true.shape[-1]))
    if null_class is not None:
        valid_labels = [x for x in valid_labels if x != null_class]
    m_micro = metrics.precision_recall_fscore_support(
        p_true,
        p_pred,
        labels=valid_labels,
        average="micro",
        zero_division=0
    )
    m_macro = metrics.precision_recall_fscore_support(
        p_true,
        p_pred,
        labels=valid_labels,
        average="macro",
        zero_division=0
    )
    m_samples = metrics.precision_recall_fscore_support(
        p_true,
        p_pred,
        labels=valid_labels,
        average="samples",
        zero_division=0
    )
    out = {
        "pr_auc_macro": metrics.average_precision_score(p_true, p_prob, average="macro"),
        "pr_auc_micro": metrics.average_precision_score(p_true, p_prob, average="micro"),
        "f1_micro": m_micro[2],
        "f1_macro": m_macro[2],
        "f1_samples": m_samples[2],
        "precision_micro": m_micro[0],
        "precision_macro": m_macro[0],
        "precision_samples": m_samples[0],
        "recall_micro": m_micro[1],
        "recall_macro": m_macro[1],
        "recall_samples": m_samples[1],
    }
    return out, prediction
