import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv('corpus.csv', sep=';')
print('file read')
def convert_str_to_int(x):
    if x == 'P':
        return 0
    if x == 'N':
        return 1
    return 2


df['tag'] = df['tag'].apply(convert_str_to_int)
X = df.drop(columns=['tag'])
y = df['tag']
_, data, _, labels = train_test_split(X, y, test_size=0.3, random_state=42)
print('train test split')
labels = list(labels)
model_name = 'dccuchile/bert-base-spanish-wwm-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
print('tokenizer loaded')
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


data['tokens'] = data['text'].apply(tokenize_text)
print('tokenized text')
def get_id(token):
    return token['input_ids']

def get_attention(token):
    return token['attention_mask']

data['input_ids'] = data['tokens'].apply(get_id)
data['attention_mask'] = data['tokens'].apply(get_attention)
input_ids = torch.tensor(list(data['input_ids'])).to(device)
attention_mask = torch.tensor(list(data['attention_mask'])).to(device)
labels = torch.tensor(labels)
print('go to train data')
train_data = TensorDataset(input_ids, attention_mask, labels)
sampler = RandomSampler(train_data)
dataloader = DataLoader(train_data, sampler=sampler, batch_size=8)
print('i dunno')

model_name = 'dccuchile/bert-base-spanish-wwm-uncased'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.load_state_dict(torch.load('/home/diwa/Documents/TDDE16_TextMining/PROJECT/model4.pt'))
#model.load_state_dict(torch.load('/home/diwa/Documents/TDDE16_TextMining/PROJECT/model3.pt', map_location=torch.device('cpu')))
print('model')
model.to(device)
model.eval()

print('model loaded')
all_predictions, all_labels = [], []

for batch in dataloader:
    print('batch', batch)
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions = np.argmax(logits, axis=1).flatten()
        #print(predictions)

        all_labels.extend(label_ids)
        all_predictions.extend(predictions)


accuracy = accuracy_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions, average='macro')
f1 = f1_score(all_labels, all_predictions, average='macro')
cm = confusion_matrix(all_labels, all_predictions)
print("Accuracy: {:.2f}, Recall: {:.2f}, F1 Score: {:.2f}".format(accuracy, recall, f1))
print("Confusion Matrix:")
print(cm)
fig, ax = plot_confusion_matrix(conf_mat = cm, figsize=(3,3), class_names = [0,1,2])
plt.show()