import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import matthews_corrcoef
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW
from transformers import AdamW, BertTokenizer
# from pytorch_pretrained_bert import  BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)  #'NVIDIA GeForce RTX 3060 Laptop GPU'

data = pd.read_csv('corpus.csv', sep=';')

def convert_str_to_int(x):
    if x == 'P':
        return 0
    if x == 'N':
        return 1
    return 2


data['tag'] = data['tag'].apply(convert_str_to_int)
X = data.drop(columns=['tag'])
y = data['tag']
df, testd, dfy, testdy = train_test_split(X, y, test_size=0.3, random_state=42)

# create corpus
# 1. convert column text to list
# 2. add cls and sep -> corpus

model_name = 'dccuchile/bert-base-spanish-wwm-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

# encode plus does:
# 1. tokenize the sentence
# 2. prepend the [CLS] token to the start
# 3. append the [SEP] token to the end
# 4. map tokens to their ids
# 5. pad or truncate the sentence to max_length
# 6. create attention masks for [PAD] tokens

# convert to tensors
def tokenize_text(text):
    # return dict of input ids and attention masks
    return tokenizer.encode_plus(text,
                                 add_special_tokens=True,  # add [CLS] and [SEP]
                                 max_length=280,
                                 pad_to_max_length=True,
                                 return_attention_mask=True,
                                 truncation=True)
                                 #return_tensors='pt')  # return pytorch tensors, if not it return list


df['tokens'] = df['text'].apply(tokenize_text)
testd['tokens'] = testd['text'].apply(tokenize_text)


def get_id(token):
    return token['input_ids']

def get_attention(token):
    return token['attention_mask']


df['input_ids'] = df['tokens'].apply(get_id)
df['attention_mask'] = df['tokens'].apply(get_attention)
testd['input_ids'] = testd['tokens'].apply(get_id)
testd['attention_mask'] = testd['tokens'].apply(get_attention)

input_ids = list(df['input_ids'])
attention_mask = list(df['attention_mask'])

train_inp, val_inp, train_labels, val_labels = train_test_split(input_ids, list(dfy), random_state=42, test_size=0.3)
train_masks, val_masks, _, _ = train_test_split(attention_mask, input_ids, random_state=42, test_size=0.3)

train_inp = torch.tensor(train_inp)
val_inp = torch.tensor(val_inp)
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
train_masks = torch.tensor(train_masks)
val_masks = torch.tensor(val_masks)


###############
batch_size = 8

train_data = TensorDataset(train_inp, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
val_data = TensorDataset(val_inp, val_masks, val_labels)
val_sampler = RandomSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

model.cuda()

# BERT finetuning parametetrs
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [{
    'params': [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
    'weight_decay_rate': 0.05
    },
    {
    'params': [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
    'weight_decay_rate': 0.0
    }
]
optimizer = AdamW(params=[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
                  correct_bias=False,
                  weight_decay=0.05,
                  lr=1e-5)

def flat_accuracy(preds, labels):
    #pred_flat = np.argmax(preds,axis=2).flatten()
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

train_loss_set = []
epochs = 6

for _ in trange(epochs, desc='Epoch'):
    ## TRAINING
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0,0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        # forward pass
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        train_loss_set.append(loss.item())
        # backward pass
        loss.backward()
        # update parameters and take a step using the computed grad
        optimizer.step()
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
    print("Train loss: {}".format(tr_loss/nb_tr_steps))

    ## validation
    model.eval()
    eval_loss, eval_accuracy = 0,0
    nb_eval_steps, nb_eval_examples = 0,0
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
    print("Validation accuracy:{}".format(eval_accuracy/nb_eval_steps))

# plots
plt.figure(figsize=(15,8))
plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)
plt.show()

#model.save_model("/home/diwa/Documents/TDDE16_TextMining/PROJECT/my_model")
torch.save(model.state_dict(), '/home/diwa/Documents/TDDE16_TextMining/PROJECT/model4.pt')

