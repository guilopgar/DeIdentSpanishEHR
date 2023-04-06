#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Model hyper-parameters:
- For both models, the hyper-parameters are:
    - batch size: 64
    - mask_value: -99
    - epochs: 500
    - early_stop: 75
    - threshold: 0.5
    - null_class: None
    - sequence length: 400
"""

# In[ ]:


# Possible values for the hyper-parameters
MODEL_NAME = "bi-lstm"
DA_FLAG = 1  # 1: add augmnented docs; 0: only train on original docs
BATCH_SIZE = 64
MASK_VALUE = -99
EPOCHS = 500
EARLY_STOP = 75
THRESHOLD = 0.5
NULL_CLASS = None
SEQ_LEN = 400
CUDA_GPU_ID = "2"

# In[ ]:


import sys

if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[-10]
    DA_FLAG = int(sys.argv[-9])
    BATCH_SIZE = int(sys.argv[-8])
    MASK_VALUE = int(sys.argv[-7])
    EPOCHS = int(sys.argv[-6])
    EARLY_STOP = int(sys.argv[-5])
    THRESHOLD = float(sys.argv[-4])
    NULL_CLASS = str(sys.argv[-3])
    SEQ_LEN = int(sys.argv[-2])
    CUDA_GPU_ID = sys.argv[-1]

# In[ ]:


# Sanity check
print("MODEL_NAME:", MODEL_NAME)
print("DA_FLAG:", DA_FLAG)
print("BATCH_SIZE:", BATCH_SIZE)
print("MASK_VALUE:", MASK_VALUE)
print("EPOCHS:", EPOCHS)
print("EARLY_STOP:", EARLY_STOP)
print("THRESHOLD:", THRESHOLD)
print("NULL_CLASS:", NULL_CLASS)
print("SEQ_LEN:", SEQ_LEN)
print("CUDA_GPU_ID:", CUDA_GPU_ID)

# In[ ]:


import os

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_GPU_ID

# In[ ]:


utils_path = "./"
model_root_path = "../../../../NLP/models/" + MODEL_NAME + "/"
dataset_path = "./datasets/"
corpus_path = dataset_path + "deident_strat_split/"
ss_corpus_path = dataset_path + "all_files-SSplit-text/"


# In[ ]:


import tensorflow as tf

# Auxiliary components
import sys
import random

sys.path.insert(0, utils_path)
from nlp_utils import *

# Hyper-parameters
subtask = 'norm'
subtask_ann = subtask + "-iob_disc"
text_col = "raw_text"

EVAL_STRATEGY = "word-prod"
mention_strat = "prod"

NEU_RNN1 = 128
DROP_RNN1 = 0.3
NEU_RNN2 = 32
DROP_RNN2 = 0.2
NEU_DENSE = 16

ROUND_N = 4

# Random seed
random_seed = 12345
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# In[ ]:


import json

with open(corpus_path + "etiquetas_sin_ec.json") as json_file:
    phi_types = json.load(json_file)

# ## Load text

# ### Training

# In[ ]:


train_path = corpus_path + "train/"
if DA_FLAG == 1:
    train_files = [f for f in os.listdir(train_path) if os.path.isfile(train_path + f) and f.split('.')[-1] == "txt"]
else:
    train_files = [f for f in os.listdir(train_path) if
                   os.path.isfile(train_path + f) and (len(f.split('_')) == 1) and f.split('.')[-1] == "txt"]

train_data = load_text_files(train_files, train_path)
df_text_train = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in train_files], 'raw_text': train_data})

# In[ ]:


print(len(set(df_text_train['doc_id'])))

# ### Development

# In[ ]:


dev_path = corpus_path + "val/"
if DA_FLAG == 1:
    dev_files = [f for f in os.listdir(dev_path) if os.path.isfile(dev_path + f) and f.split('.')[-1] == "txt"]
else:
    dev_files = [f for f in os.listdir(dev_path) if
                 os.path.isfile(dev_path + f) and (len(f.split('_')) == 1) and f.split('.')[-1] == "txt"]
dev_data = load_text_files(dev_files, dev_path)
df_text_dev = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in dev_files], 'raw_text': dev_data})

# In[ ]:


print(len(set(df_text_dev['doc_id'])))

# ### Test

# In[ ]:


test_path = corpus_path + "test/"
test_files = [f for f in os.listdir(test_path) if os.path.isfile(test_path + f) and f.split('.')[-1] == "txt"]
test_data = load_text_files(test_files, test_path)
df_text_test = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in test_files], 'raw_text': test_data})

# In[ ]:


print(len(set(df_text_test['doc_id'])))

# ## Process annotations

# ### Training

# In[ ]:


if DA_FLAG == 1:
    train_ann_files = [train_path + f for f in os.listdir(train_path) if
                       os.path.isfile(train_path + f) and f.split('.')[-1] == "ann"]
else:
    train_ann_files = [train_path + f for f in os.listdir(train_path) if
                       os.path.isfile(train_path + f) and (len(f.split('_')) == 1) and f.split('.')[-1] == "ann"]

# In[ ]:


df_codes_train_ner = process_de_ident_ner(train_ann_files)

# In[ ]:


df_codes_train_ner = df_codes_train_ner[df_codes_train_ner['type'].apply(lambda x: x in phi_types.keys())]
df_codes_train_ner["type"] = df_codes_train_ner["type"].apply(lambda x: phi_types[x])

# In[ ]:


df_codes_train_ner.rename(columns={"type": "code"}, inplace=True)

# In[ ]:


df_codes_train_ner['start'] = df_codes_train_ner['location'].apply(lambda x: int(x.split(' ')[0]))
df_codes_train_ner['end'] = df_codes_train_ner['location'].apply(lambda x: int(x.split(' ')[-1]))

# In[ ]:


df_codes_train_ner.sort_values(["doc_id", "start", "end"], inplace=True)

# In[ ]:


assert ~df_codes_train_ner[["doc_id", "start", "end"]].duplicated().any()

# In[ ]:


print(len(set(df_codes_train_ner['doc_id'])))

# ### Development

# In[ ]:


if DA_FLAG == 1:
    dev_ann_files = [dev_path + f for f in os.listdir(dev_path) if
                     os.path.isfile(dev_path + f) and f.split('.')[-1] == "ann"]
else:
    dev_ann_files = [dev_path + f for f in os.listdir(dev_path) if
                     os.path.isfile(dev_path + f) and (len(f.split('_')) == 1) and f.split('.')[-1] == "ann"]

# In[ ]:


df_codes_dev_ner = process_de_ident_ner(dev_ann_files)

# In[ ]:


df_codes_dev_ner = df_codes_dev_ner[df_codes_dev_ner['type'].apply(lambda x: x in phi_types.keys())]
df_codes_dev_ner["type"] = df_codes_dev_ner["type"].apply(lambda x: phi_types[x])

# In[ ]:


df_codes_dev_ner.rename(columns={"type": "code"}, inplace=True)

# In[ ]:


df_codes_dev_ner['start'] = df_codes_dev_ner['location'].apply(lambda x: int(x.split(' ')[0]))
df_codes_dev_ner['end'] = df_codes_dev_ner['location'].apply(lambda x: int(x.split(' ')[-1]))

# In[ ]:


df_codes_dev_ner.sort_values(["doc_id", "start", "end"], inplace=True)

# In[ ]:


assert ~df_codes_dev_ner[["doc_id", "start", "end"]].duplicated().any()

# In[ ]:


print(len(set(df_codes_dev_ner['doc_id'])))

# ### Test

# In[ ]:


test_ann_files = [test_path + f for f in os.listdir(test_path) if
                  os.path.isfile(test_path + f) and f.split('.')[-1] == "ann"]

# In[ ]:


df_codes_test_ner = process_de_ident_ner(test_ann_files)

# In[ ]:


df_codes_test_ner = df_codes_test_ner[df_codes_test_ner['type'].apply(lambda x: x in phi_types.keys())]
df_codes_test_ner["type"] = df_codes_test_ner["type"].apply(lambda x: phi_types[x])

# In[ ]:


df_codes_test_ner.rename(columns={"type": "code"}, inplace=True)

# In[ ]:


df_codes_test_ner['start'] = df_codes_test_ner['location'].apply(lambda x: int(x.split(' ')[0]))
df_codes_test_ner['end'] = df_codes_test_ner['location'].apply(lambda x: int(x.split(' ')[-1]))

# In[ ]:


df_codes_test_ner.sort_values(["doc_id", "start", "end"], inplace=True)

# In[ ]:


assert ~df_codes_test_ner[["doc_id", "start", "end"]].duplicated().any()

# In[ ]:


print(len(set(df_codes_test_ner['doc_id'])))

# ## Creation of annotated sequences

# In[ ]:


train_dev_codes = sorted(set(df_codes_train_ner["code"].values))

# In[ ]:


print(len(train_dev_codes))

# In[ ]:


# Create label encoders as dict (more computationally efficient)
i = 0
lab_encoder = {}
lab_decoder = {}
for iob in ["B", "I"]:
    for code in train_dev_codes:
        lab_encoder[iob + "-" + code] = i
        lab_decoder[i] = iob + "-" + code
        i += 1
lab_encoder["O"] = i
lab_decoder[i] = "O"

# In[ ]:


print(len(lab_encoder), len(lab_decoder))

# In[ ]:


# Text classification (later ignored)


# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer

mlb_encoder = MultiLabelBinarizer()
mlb_encoder.fit([train_dev_codes])

# In[ ]:


# Sentence-Split data


# In[ ]:


ss_files = [f for f in os.listdir(ss_corpus_path) if os.path.isfile(ss_corpus_path + f)]
ss_dict_corpus = load_ss_files(ss_files, ss_corpus_path)

# In[ ]:


# Sanity check


# In[ ]:


assert len(set(df_codes_train_ner["doc_id"]).intersection(set(df_codes_dev_ner["doc_id"]))) == 0
assert len(set(df_codes_train_ner["doc_id"]).intersection(set(df_codes_test_ner["doc_id"]))) == 0
assert len(set(df_codes_dev_ner["doc_id"]).intersection(set(df_codes_test_ner["doc_id"]))) == 0

# ### Training corpus

# Only texts with NER annotations are considered:

# In[ ]:


train_doc_list = sorted(set(df_codes_train_ner["doc_id"]))

# In[ ]:


train_gen = process_data(df_text_train, df_codes_train_ner, train_doc_list, [lab_encoder], SEQ_LEN)


# ### Development corpus

# Only texts with NER annotations are considered:

# In[ ]:


dev_doc_list = sorted(set(df_codes_dev_ner["doc_id"]))

# In[ ]:


dev_gen = process_data(df_text_dev, df_codes_dev_ner, dev_doc_list, [lab_encoder], SEQ_LEN)


# ### Test corpus

# All texts are considered:

# In[ ]:


test_doc_list = sorted(set(df_text_test["doc_id"]))

# In[ ]:


test_gen = process_data(df_text_test, df_codes_test_ner, test_doc_list, [lab_encoder], SEQ_LEN)


# ## Create model

# In[ ]:


# Set memory growth


# In[ ]:


physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

# In[ ]:

from tensorflow.keras import models as k_models
from tensorflow.keras import layers as k_layers
from tensorflow.keras import metrics as k_metrics
import galennlp_keras_crf as _crf

iob_num_labels = len(lab_encoder)

if MODEL_NAME.split('_')[0] == 'bi-lstm' or MODEL_NAME.split('-')[0] == '2-bi-lstm':
    inputs = k_layers.Input(shape=(None, n_features))
    masked = k_layers.Masking(mask_value=MASK_VALUE)(inputs)
    rnl = k_layers.Bidirectional(k_layers.LSTM(NEU_RNN1, recurrent_dropout=DROP_RNN1, return_sequences=True))(masked)

    if MODEL_NAME.split('-')[0] == '2-bi-lstm':
        rnl = k_layers.Bidirectional(k_layers.LSTM(NEU_RNN2, recurrent_dropout=DROP_RNN2, return_sequences=True))(rnl)

    dense = k_layers.TimeDistributed(k_layers.Dense(NEU_DENSE, activation="relu"))(rnl)
    output = k_layers.TimeDistributed(k_layers.Dense(iob_num_labels, activation="sigmoid"))(dense)
    model = k_models.Model(inputs, output)

    model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=[
            k_metrics.CategoricalAccuracy(),
            k_metrics.AUC(curve="PR", name="auc_pr"),
            k_metrics.Precision(),
            k_metrics.Recall()
        ]
    )

elif MODEL_NAME.split('-')[0] == 'bi-lstm-crf':
    inputs = k_layers.Input(shape=(None, n_features))
    masked = k_layers.Masking(mask_value=mask_value)(inputs)
    rnl = k_layers.Bidirectional(k_layers.LSTM(128, recurrent_dropout=0.3, return_sequences=True))(masked)
    dense = k_layers.TimeDistributed(k_layers.Dense(n_classes))(rnl)

    crf = _crf.CRF(n_classes)
    output = crf(dense)
    model = k_models.Model(inputs, output)

    model.compile(
        optimizer=k_optimizers.Adam(learning_rate=0.0005),
        loss=crf.loss,
        metrics=[crf.accuracy]
    )
else:
    model = None

# In[ ]:


print(model.summary())

# In[ ]:


print(model.input)

# In[ ]:


print(model.output)

# In[ ]:


df_train_gs = df_codes_train_ner.rename(columns={'doc_id': 'clinical_case', 'code': 'code_gs'})
df_dev_gs = df_codes_dev_ner.rename(columns={'doc_id': 'clinical_case', 'code': 'code_gs'})
df_test_gs = df_codes_test_ner.rename(columns={'doc_id': 'clinical_case', 'code': 'code_gs'})

# In[ ]:


import time


start_time = time.time()

model = train_rnn_model(model=model, gen_train=train_gen, gen_valid=dev_gen,
                        batch_size=BATCH_SIZE, epochs=EPOCHS, patience=EARLY_STOP, null_class=NULL_CLASS,
                        threshold=THRESHOLD, model_path=model_root_path)

end_time = time.time()

# In[ ]:


print("--- %s minutes ---" % ((end_time - start_time) / 60))

# ## Evaluation

# ### Test

# In[ ]:


_, test_preds = evaluate_model(model=model, data_gen=test_gen, null_class=NULL_CLASS, threshold=THRESHOLD)

# In[ ]:


df_pred_test = ner_preds_brat_format(doc_list=test_doc_list, fragments=test_frag,
                                     preds=[test_preds],
                                     start_end=test_start_end_frag, word_id=test_word_id,
                                     lab_decoder_list=[lab_decoder],
                                     df_text=df_text_test,
                                     text_col=text_col, strategy=EVAL_STRATEGY, subtask=subtask_ann,
                                     mention_strat=mention_strat)

# In[ ]:


df_pred_test = df_pred_test[["clinical_case", "location", "code_pred"]]

# In[ ]:


df_res_test, p_test, r_test, f1_test = calculate_anon_metrics(gs=df_test_gs, pred=df_pred_test)

# In[ ]:


# Micro-average
p_test_micro, r_test_micro, f1_test_micro = calculate_anon_single_metrics(gs=df_test_gs, pred=df_pred_test)

# In[ ]:


# NER evaluation
df_test_gs_ner = df_test_gs.copy()
df_test_gs_ner["code_gs"] = "X"

df_pred_test_ner = df_pred_test.copy()
df_pred_test_ner["code_pred"] = "X"

p_test_ner, r_test_ner, f1_test_ner = calculate_anon_single_metrics(gs=df_test_gs_ner, pred=df_pred_test_ner)

# In[ ]:


train_label_freq = df_train_gs.code_gs.value_counts()
dev_label_freq = df_dev_gs.code_gs.value_counts()
test_label_freq = df_test_gs.code_gs.value_counts()
df_res_test['Support'] = [test_label_freq[label] for label in df_res_test.index]
df_res_test['Train+Dev Support'] = [train_label_freq[label] + dev_label_freq[label] for label in df_res_test.index]
df_res_test = pd.concat((df_res_test, pd.DataFrame({
    'P': {'macro-avg': p_test, 'micro-avg': p_test_micro, 'exact': p_test_ner},
    'R': {'macro-avg': r_test, 'micro-avg': r_test_micro, 'exact': r_test_ner},
    'F1': {'macro-avg': f1_test, 'micro-avg': f1_test_micro, 'exact': f1_test_ner},
    'Support': {'macro-avg': None, 'micro-avg': None, 'exact': None},
    'Train+Dev Support': {'macro-avg': None, 'micro-avg': None, 'exact': None}
})))

# Round decimals
for metric in ['P', 'R', 'F1']:
    df_res_test[metric] = df_res_test[metric].apply(lambda x: round(x, ROUND_N))

# In[ ]:


print("Test results:")
print(df_res_test)