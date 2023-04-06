#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Model hyper-parameters:

- For the "base" version of both models, the hyper-parameters are:
    - batch size: 24
    - learning rate (LR): 8e-6
    - sequence length (SEQ_LEN): 256
    - epochs: 60

- For the "large" version of both models:
    - batch size: 24
    - learning rate (LR): 3e-6
    - sequence length (SEQ_LEN): 256
    - epochs: 40
    
"""


# In[ ]:


# Possible values for the hyper-parameters
MODEL_NAME = "xlm-roberta-base"
DA_FLAG = 1 # 1: add augmnented docs; 0: only train on original docs
BATCH_SIZE = 24
EPOCHS = 60
SEQ_LEN = 256
LR = 8e-6
CUDA_GPU_ID = "3"


# In[ ]:


import sys
if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[-7]
    DA_FLAG = int(sys.argv[-6])
    BATCH_SIZE = int(sys.argv[-5])
    EPOCHS = int(sys.argv[-4])
    SEQ_LEN = int(sys.argv[-3])
    LR = float(sys.argv[-2])
    CUDA_GPU_ID = sys.argv[-1]


# In[ ]:


# Sanity check
print("MODEL_NAME:", MODEL_NAME)
print("DA_FLAG:", DA_FLAG)
print("BATCH_SIZE:", BATCH_SIZE)
print("EPOCHS:", EPOCHS)
print("SEQ_LEN:", SEQ_LEN)
print("LR:", LR)
print("CUDA_GPU_ID:", CUDA_GPU_ID)


# In[ ]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_GPU_ID


# In[ ]:


utils_path = "./"
model_root_path = "../../../../NLP/models/"
dataset_path = "./datasets/"
corpus_path = dataset_path + "deident_strat_split/"
ss_corpus_path = dataset_path + "all_files-SSplit-text/"


# In[ ]:


from transformers import BertTokenizerFast, XLMRobertaTokenizerFast, RobertaTokenizerFast

if MODEL_NAME == 'xlm-roberta-base':
    model_path = model_root_path + "XLM-R/pytorch/" + MODEL_NAME
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)

elif MODEL_NAME == 'xlm-roberta-large':
    model_path = model_root_path + "XLM-R/pytorch/" + MODEL_NAME
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
elif MODEL_NAME == 'roberta-base-bne':
    model_path = model_root_path + "RoBERTa/pytorch/" + MODEL_NAME
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
elif MODEL_NAME == 'roberta-large-bne':
    model_path = model_root_path + "RoBERTa/pytorch/" + MODEL_NAME
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
else:
    print("ERROR: NO AVAILABLE MODEL!!")
    print()


# In[ ]:


import tensorflow as tf

# Auxiliary components
import sys
sys.path.insert(0, utils_path)
from nlp_utils import *


# Hyper-parameters
type_tokenizer = "transformers"

subtask = 'norm'
subtask_ann = subtask + "-iob_disc"
text_col = "raw_text"

GREEDY = True
IGNORE_VALUE = -100
code_strat = 'o'
ANN_STRATEGY = "word-all"
EVAL_STRATEGY = "word-prod"
mention_strat = "prod"
LOGITS = False

ROUND_N = 4

random_seed = 0
tf.random.set_seed(random_seed)


# In[ ]:


import json
with open(corpus_path + "etiquetas_sin_ec.json") as json_file:
    phi_types = json.load(json_file)


# ## Load text

# ### Training

# In[ ]:


train_path = corpus_path + "train/"
if DA_FLAG == 1:
    train_files = [f for f in os.listdir(train_path) if os.path.isfile(train_path + f) and                    f.split('.')[-1] == "txt"]
else:
    train_files = [f for f in os.listdir(train_path) if os.path.isfile(train_path + f) and                    (len(f.split('_')) == 1) and f.split('.')[-1] == "txt"]

train_data = load_text_files(train_files, train_path)
df_text_train = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in train_files], 'raw_text': train_data})


# In[ ]:


print(len(set(df_text_train['doc_id'])))


# ### Development

# In[ ]:


dev_path = corpus_path + "val/"
if DA_FLAG == 1:
    dev_files = [f for f in os.listdir(dev_path) if os.path.isfile(dev_path + f) and                  f.split('.')[-1] == "txt"]
else:
    dev_files = [f for f in os.listdir(dev_path) if os.path.isfile(dev_path + f) and                  (len(f.split('_')) == 1) and f.split('.')[-1] == "txt"]
dev_data = load_text_files(dev_files, dev_path)
df_text_dev = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in dev_files], 'raw_text': dev_data})


# In[ ]:


print(len(set(df_text_dev['doc_id'])))


# ### Test

# In[ ]:


test_path = corpus_path + "test/"
test_files = [f for f in os.listdir(test_path) if os.path.isfile(test_path + f) and               f.split('.')[-1] == "txt"]
test_data = load_text_files(test_files, test_path)
df_text_test = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in test_files], 'raw_text': test_data})


# In[ ]:


print(len(set(df_text_test['doc_id'])))


# ## Process annotations

# ### Training

# In[ ]:


if DA_FLAG == 1:
    train_ann_files = [train_path + f for f in os.listdir(train_path) if os.path.isfile(train_path + f) and                        f.split('.')[-1] == "ann"]
else:
    train_ann_files = [train_path + f for f in os.listdir(train_path) if os.path.isfile(train_path + f) and                        (len(f.split('_')) == 1) and f.split('.')[-1] == "ann"]


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
    dev_ann_files = [dev_path + f for f in os.listdir(dev_path) if os.path.isfile(dev_path + f) and                      f.split('.')[-1] == "ann"]
else:
    dev_ann_files = [dev_path + f for f in os.listdir(dev_path) if os.path.isfile(dev_path + f) and                      (len(f.split('_')) == 1) and f.split('.')[-1] == "ann"]


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


test_ann_files = [test_path + f for f in os.listdir(test_path) if os.path.isfile(test_path + f) and                   f.split('.')[-1] == "ann"]


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


train_ind, train_att, train_type, train_y, train_text_y, train_frag, train_start_end_frag,                 train_word_id = ss_create_input_data_ner(df_text=df_text_train, text_col=text_col, 
                                    df_ann=df_codes_train_ner, df_ann_text=df_codes_dev_ner, # ignore text-level 
                                    doc_list=train_doc_list, ss_dict=ss_dict_corpus,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat, ss_sep=True)


# In[ ]:


train_y = train_y[0]


# ### Development corpus

# Only texts with NER annotations are considered:

# In[ ]:


dev_doc_list = sorted(set(df_codes_dev_ner["doc_id"]))


# In[ ]:


dev_ind, dev_att, dev_type, dev_y, dev_text_y, dev_frag, dev_start_end_frag,                 dev_word_id = ss_create_input_data_ner(df_text=df_text_dev, text_col=text_col, 
                                    df_ann=df_codes_dev_ner, df_ann_text=df_codes_train_ner, # ignore text-level 
                                    doc_list=dev_doc_list, ss_dict=ss_dict_corpus,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat, ss_sep=True)


# In[ ]:


dev_y = dev_y[0]


# ### Test corpus

# All texts are considered:

# In[ ]:


test_doc_list = sorted(set(df_text_test["doc_id"]))


# In[ ]:


test_ind, test_att, test_type, test_y, test_text_y, test_frag, test_start_end_frag,                 test_word_id = ss_create_input_data_ner(df_text=df_text_test, text_col=text_col, 
                                    df_ann=df_codes_dev_ner, df_ann_text=df_codes_dev_ner, # ignore text-level 
                                    doc_list=test_doc_list, ss_dict=ss_dict_corpus,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat, ss_sep=True)


# In[ ]:


test_y = test_y[0]


# ### Training & Development corpus
# 
# We merge the previously generated datasets:

# In[ ]:


# Indices
train_dev_ind = np.concatenate((train_ind, dev_ind))


# In[ ]:


print(train_dev_ind.shape)


# In[ ]:


# Attention
train_dev_att = np.concatenate((train_att, dev_att))


# In[ ]:


print(train_dev_att.shape)


# In[ ]:


train_dev_y = np.concatenate((train_y, dev_y))


# In[ ]:


print(train_dev_y.shape)


# ## Fine-tuning

# In[ ]:


# Set memory growth


# In[ ]:


physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)


# In[ ]:


from transformers import TFBertForTokenClassification, TFXLMRobertaForTokenClassification, TFRobertaForTokenClassification 

if MODEL_NAME.split('_')[0] == 'xlmr':
    model = TFXLMRobertaForTokenClassification.from_pretrained(model_path, from_pt=True)

elif MODEL_NAME.split('-')[0] == 'roberta':
    model = TFRobertaForTokenClassification.from_pretrained(model_path, from_pt=True)


# In[ ]:


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.initializers import GlorotUniform

iob_num_labels = len(lab_encoder)

input_ids = Input(shape=(SEQ_LEN,), name='input_ids', dtype='int64')
attention_mask = Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int64')

out_seq = model.layers[0](input_ids=input_ids, attention_mask=attention_mask)[0] # take the output sub-token sequence 

# IOB-2
out_iob = Dense(units=iob_num_labels, kernel_initializer=GlorotUniform(seed=random_seed))(out_seq) # Multi-class classification 
out_iob_model = Activation(activation='softmax', name='iob_output')(out_iob)

model = Model(inputs=[input_ids, attention_mask], outputs=out_iob_model)


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


import tensorflow_addons as tfa
import time

optimizer = tfa.optimizers.RectifiedAdam(learning_rate=LR)
loss = {'iob_output': TokenClassificationLoss(from_logits=LOGITS, ignore_val=IGNORE_VALUE)}
loss_weights = {'iob_output': 1}
model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

start_time = time.time()

history = model.fit(x={'input_ids': train_dev_ind, 'attention_mask': train_dev_att}, 
                    y={'iob_output': train_dev_y}, 
                    batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True,
                    validation_data=None, 
                    verbose=1)

end_time = time.time()


# In[ ]:


print("--- %s minutes ---" % ((end_time - start_time) / 60))


# ## Evaluation

# ### Test

# In[ ]:


test_preds = model.predict({'input_ids': test_ind, 'attention_mask': test_att})


# In[ ]:


df_pred_test = ner_preds_brat_format(doc_list=test_doc_list, fragments=test_frag, 
                                    preds=[test_preds],
                                    start_end=test_start_end_frag, word_id=test_word_id, 
                                    lab_decoder_list=[lab_decoder], 
                                    df_text=df_text_test, 
                                    text_col=text_col, strategy=EVAL_STRATEGY, subtask=subtask_ann, mention_strat=mention_strat)


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

