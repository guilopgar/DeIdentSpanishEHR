#!/usr/bin/env python
import os
import pandas as pd
from typing import (
    Iterator,
)

import numpy as np
from pathlib import Path
from tqdm import tqdm

from galennlp_fasttext.readonly import ReadonlyFasttext
from galennlp_tools.flags import WordFlags
from galennlp_fragments.fragments import IndexedLine
from galennlp_corpus.paragraph.blocks import BlockJoiner
from galennlp_tokens import Tokenizer
from galennlp_corpus.corpus import (
    Corpus,
)
from auxiliar import tag_format
from galennlp_tokens.brat.utils import TokenTransformer


def check_dir(directory: str):
    """
    Checks if a path does not exist, and creates it.
    """
    try:
        assert os.path.exists(directory)
    except AssertionError:
        os.makedirs(directory)


not_bio_dict = {"O": 0,
                "B-CENTRO": 1,
                "I-CENTRO": 2,
                "B-CONTACTO": 3,
                "B-DIRECCION": 4,
                "I-DIRECCION": 5,
                "B-HISTORIA": 6,
                "B-IDENT": 7,
                "I-IDENT": 8,
                "B-PERSONA": 9,
                "I-PERSONA": 10,
                "B-REFERENCIA": 11,
                "B-UBICACION": 12,
                "I-UBICACION": 13
                }


def count_docs_with_entities(args, fold):
    corpus = os.listdir(args['path'] + args['folder_csv'])
    files = []
    files_DA = []

    for i in corpus:
        if "_0" in i:
            files_DA.append(i)
        if "_0" not in i:
            files.append(i)
            files_DA.append(i)

    total = len(files)
    total_DA = len(files_DA)

    with_ne = 0
    without_ne = 0

    for i, filename in enumerate(tqdm(files, desc="file")):
        ann_file = pd.read_csv(args['path'] + args['folder_csv'] + filename, delimiter=";")
        nrows = len(ann_file.index)
        for ent in ann_file.iterrows():
            print(ent)
        with_ne += 1 if nrows != 0 else 0
        without_ne += 1 if nrows == 0 else 0

    print("Fold:                 " + fold)
    print("Nº Files:             " + str(total))
    print("Nº Files with NEs:    " + str(with_ne))
    print("Nº Files without NEs: " + str(without_ne) + "\n")

    if fold != 'test':
        for i, filename in enumerate(tqdm(files_DA, desc="file")):
            ann_file = pd.read_csv(args['path'] + args['folder_csv'] + filename, delimiter=";")
            nrows = len(ann_file.index)
            with_ne += 1 if nrows != 0 else 0
            without_ne += 1 if nrows == 0 else 0

        print("\nFold:                 " + fold + "_DA")
        print("Nº Files:             " + str(total_DA))
        print("Nº Files with NEs:    " + str(with_ne))
        print("Nº Files without NEs: " + str(without_ne) + "\n")


class TextTagger:

    def __init__(self, fasttext: Path, wordvectors: Path, ngrams: Path):
        self.TOK = Tokenizer()
        self.JOINER = BlockJoiner()
        self.transformer = TokenTransformer([], {})
        # word embedding
        self.kv = ReadonlyFasttext(str(fasttext), str(wordvectors), str(ngrams))
        # dics
        self.labels_dic = {
            "B-CENTRO": 0, "I-CENTRO": 0, "B-CONTACTO": 0,
            "B-DIRECCION": 0, "I-DIRECCION": 0, "B-HISTORIA": 0,
            "B-IDENT": 0, "I-IDENT": 0, "B-PERSONA": 0, "I-PERSONA": 0,
            "B-REFERENCIA": 0, "B-UBICACION": 0, "I-UBICACION": 0,
            "O": 0
        }
        self.ann_dic = {
            "B-CENTRO": 0, "I-CENTRO": 0, "B-CONTACTO": 0,
            "B-DIRECCION": 0, "I-DIRECCION": 0, "B-HISTORIA": 0,
            "B-IDENT": 0, "I-IDENT": 0, "B-PERSONA": 0, "I-PERSONA": 0,
            "B-REFERENCIA": 0, "B-UBICACION": 0, "I-UBICACION": 0,
            "O": 0
        }
        # output
        self.num_class = len(self.ann_dic)
        self.dict_tags = not_bio_dict
        self.classes = [0] * self.num_class
        # idx + class + flags + tags + embedding
        self.tok_size = 4 + 12 + 400
        self.vector_size = 1 + self.num_class + self.tok_size
        self.aux_info = 4
        self.contador = 0

    def index_lines(self, text: str) -> Iterator[IndexedLine]:
        lines = text.splitlines(keepends=True)
        return self.JOINER.read_document(lines)

    def extract_features(self, tok_line: IndexedLine, text: str, idx_line: int, ann_file):
        """explore lines in document and write tokens"""
        # Numpy array with all line tokenize
        tok_vector = np.empty(shape=(0, self.vector_size))
        tok_indexed = np.empty(shape=(0, self.aux_info), dtype=object)

        # TOKEN TRANSFORM
        tok_line = list(self.transformer.transform(self.TOK.parse(tok_line)))

        set_lab = set(())
        maybe_sep = False

        for (old_tok, new_tok) in tok_line:
            # Aux Tok_vector
            tok_vector_aux = np.empty(shape=(1, self.vector_size))
            tok_start = new_tok.fragments[0].document_chunk.start
            tok_end = tok_start + new_tok.fragments[0].document_chunk.size

            id_true, lab_true = ['O', 'O']

            # Get lab from the ann if it exists.
            for index, row in ann_file.iterrows():
                if index not in set_lab:
                    if len(new_tok.fragments) == 2:
                        tok_start = new_tok.fragments[1].document_chunk.start
                        tok_end = tok_start + new_tok.fragments[1].document_chunk.size

                    if row['start'] <= tok_start < row['end']:
                        # print(new_tok)
                        # print(row['text'])

                        id_true = row['id']

                        if maybe_sep and new_tok.token != "/":
                            if row['en'][2:] not in "CONTACTO|REFERENCIA|HISTORIA":
                                lab_true = 'I-' + row['en'][2:]
                            else:
                                lab_true = row['en']
                        elif new_tok.token != "/":
                            lab_true = row['en']
                            self.labels_dic[lab_true] += 1

                        if tok_end == row['end']:
                            set_lab.add(index)
                            self.contador += 1
                            maybe_sep = False
                        else:
                            maybe_sep = True
                        break

            # Add new Token
            aux_ = np.array([old_tok[0], new_tok[1], id_true, lab_true], dtype=object).reshape([1, self.aux_info])
            tok_indexed = np.concatenate((tok_indexed, aux_), axis=0)

            # Get entity class.
            aux_classes = [0] * self.num_class
            aux_classes[int(self.dict_tags[lab_true])] = 1

            # OUTPUT
            lim_inf = 0
            lim_sup = 1 + self.num_class
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
            tok_vector_aux[0, lim_inf:] = self.kv[new_tok.token.value]

            # UPDATE
            tok_vector = np.concatenate((tok_vector, tok_vector_aux), axis=0)

        return tok_vector, tok_indexed

    def predict_brat(self, text: str, folder_csv: str, file_tok: str):
        ann_file = pd.read_csv(folder_csv, delimiter=";")
        self.contador = 0
        for index, row in ann_file.iterrows():
            self.ann_dic[row['en']] += 1

        numpy_tok = np.empty(shape=(0, self.vector_size))
        numpy_ind = np.empty(shape=(0, self.aux_info), dtype=object)

        # File to save print
        idx_line = 0

        for line in self.index_lines(text):
            # List all tokens in line.
            tokens = []

            # Update Brat with tokens in line.
            tok_line = list(self.TOK.parse(line))
            tokens.append(tok_line)

            # Get tokens numpy vector.
            tok_vector, tok_indexed = self.extract_features(line, text, idx_line, ann_file)
            idx_line += 1

            numpy_tok = np.concatenate((numpy_tok, tok_vector), axis=0)
            numpy_ind = np.concatenate((numpy_ind, tok_indexed), axis=0)

        np.save(file_tok + '.npy', np.array(numpy_tok, dtype=object))
        np.save(file_tok + '_tokens.npy', np.array(numpy_ind, dtype=object))

    def process_document(self, filename: Path, folder_csv: str, folder_tok: str):
        aux = str(filename).split("\\")
        folder_csv = folder_csv + "\\" + aux[len(aux) - 1].split(".")[0] + ".ann.csv"
        file_tok = folder_tok + "\\" + aux[len(aux) - 1].split(".")[0]

        try:
            # Open document.
            with open(filename, mode="r", encoding='utf-8') as file:
                document = "".join(file.readlines())

            # Predict and assign annotations.
            return self.predict_brat(document, folder_csv, file_tok)
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                # Open document.
                with open(filename, mode="r", encoding='utf-8-sig') as file:
                    document = "".join(file.readlines())

                # Predict and assign annotations.
                return self.predict_brat(document, folder_csv, file_tok)
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Open document.
                with open(filename, mode="r") as file:
                    document = "".join(file.readlines())

                # Predict and assign annotations.
                return self.predict_brat(document, folder_csv, file_tok)

    def all_corpus(self, args):
        corpus = Corpus(args['path'] + args['folder_text'], plain=True)
        corpus.provider = BlockJoiner()
        files = list(corpus.files())
        total = len(files)

        for i, filename in enumerate(tqdm(files, desc="file")):
            print("File " + str(i) + " de " + str(total) + ": " + str(filename))
            self.process_document(filename, args['path'] + args['folder_csv'],
                                  args['path'] + args['folder_tok'])

        with open(args['path'] + "bio-tokenization_resume.txt", 'w') as f:
            print("ANN:", file=f)
            print(self.ann_dic, file=f)
            print("LAB:", file=f)
            print(self.labels_dic, file=f)


def cli():
    fold_list = ['train', 'val', 'test']
    for fold in fold_list:
        args = {'path': "C:\\Galen\\galen_guille\\" + fold + "\\",
                'folder_text': "text\\",
                'folder_csv': "bio-csv\\",
                'folder_tok': "bio-tok\\",
                'fasttext': Path("D:/Z_Copia_Galen_C/fasttext/train_resto.skipgram.s400.w10.mc01.ftro"),
                'wordvectors': Path("D:/Z_Copia_Galen_C/fasttext/train_resto.skipgram.s400.w10.mc01.word_vectors.npy"),
                'ngrams': Path("D:/Z_Copia_Galen_C/fasttext/train_resto.skipgram.s400.w10.mc01.ngram_vectors.npy")}

        check_dir(args['path'] + args['folder_tok'])

        # project = TextTagger(args['fasttext'],
        #                      args['wordvectors'],
        #                      args['ngrams'])
        # project.all_corpus(args)
        count_docs_with_entities(args, fold)


if __name__ == "__main__":
    cli()

# Hay dos etiquetas que debido al tokenizador, han desaparecido
# C:\Galen\data\folds\09\text\S0210-48062005000800014-1.txt
#         Pasaje Petunia, 6, 41089
#         IndexedToken(token=Token{NUMEXP},
#                      fragments=(LineFragment(line=17,
#                                              document_chunk=SpanS(start=1594, size=9),
#                                              relative_chunk=SpanS(start=0, size=9)),))

# C:\Galen\data\folds\09\text\S0378-48352006000300006-1.txt
#         Av. Alcalde Rovira Roure, 80, 25007
#         IndexedToken(token=Token{NUMEXP},
#                      fragments=(LineFragment(line=25,
#                                              document_chunk=SpanS(start=2631, size=10),
#                                              relative_chunk=SpanS(start=0, size=10)),))
