#!/usr/bin/env python
# coding: utf-8

# # Import

# In[ ]:


import time
from os import listdir
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda")
from transformers import*


# # Load data & Preprocess

# In[ ]:


def to_input_format(query, doc):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    query_tokens = tokenizer.tokenize(query)
    doc_tokens = tokenizer.tokenize(doc)
    input_tokens = [cls_token] + query_tokens + [sep_token] + doc_tokens
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)[:512]
    token_type = ([0] * (2+len(query_tokens)) + [1] * (len(doc_tokens)))[:512]
    attention_mask = [1] * len(input_ids)
    if len(input_ids)<512:
        pad = 512 - len(input_ids)
        input_ids = input_ids + [0] * pad
        token_type = token_type + [1] * pad
        attention_mask = attention_mask + [0] * pad
    input_ids = np.array(input_ids).reshape((1,-1))
    token_type = np.array(token_type).reshape((1,-1))
    attention_mask = np.array(attention_mask).reshape((1,-1))
    return input_ids, token_type, attention_mask


# In[ ]:


def load_training_data(train_df):
    train_x = []
    train_y = []
    for idx,row in train_df.iterrows():
        fq = open("train/query/"+str(row.query), 'r') 
        qry = fq.read()
        fd = open("doc/"+str(row.document), 'r') 
        doc = fd.read()
        train_x.append(to_input_format(qry, doc))
        train_y.append([row.label])      
    train_x = torch.tensor(train_x, dtype=torch.long)
    train_y = torch.tensor(train_y, dtype=torch.float)
    return train_x, train_y


# In[ ]:


def load_test_data(qry_file_name, doc_file_name):
    fq = open("test/query/" + qry_file_name, 'r') 
    qry = fq.read()
    fd = open("doc/" + doc_file_name, 'r') 
    doc = fd.read()
    test_x = to_input_format(qry, doc)
    test_x = torch.tensor(test_x, dtype=torch.long)
    return test_x


# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_length=512)

# training data 
pos = pd.read_csv('train/Pos.txt', sep=" ", header=None)
neg = pd.read_csv('train/Neg.txt', sep=" ", header=None)
train_df = pd.concat([pos,neg])
train_df.columns = ['query','document','label']
train_x, train_y = load_training_data(train_df)

# testing data file name list
docs = listdir('doc') 
qrys = listdir("test/query")


# # Build Model

# In[ ]:


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)


# # Train

# In[ ]:


optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()
model.to(device)
# model.load_state_dict(torch.load('model.pkl'))
model.train()


# In[ ]:


for epoch in range(1000):   
    print("epoch:",epoch)
    l=0
    for i in range(len(train_x)):   
        (input_ids, token_type, attention_mask) = train_x[i]
        labels = train_y[i]
        input_ids = input_ids.cuda()
        token_type = token_type.cuda()
        attention_mask = attention_mask.cuda()
        labels = labels.cuda()
        _,outputs = model(input_ids=input_ids, token_type_ids=token_type, 
                        attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs.squeeze(-1), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        l+=loss
    print("avg loss:",l/len(train_x))
    torch.save(model.state_dict(), 'model.pkl')


# # Test

# In[ ]:


model.eval()


# In[ ]:


fname = "./www.txt"
f = open(fname, 'w')
f.write("Query,RetrievedDocuments\n")  

for q in range(len(qrys)): 
    qry_file_name=qrys[q]
    f.write(qry_file_name + ",") 
    sim = np.zeros(len(docs))
    for j in range(len(docs)):
        doc_file_name=docs[j]
        test_x = load_test_data(qry_file_name, doc_file_name)
        input_ids, token_type, attention_mask = test_x
        input_ids = input_ids.cuda()
        token_type = token_type.cuda()
        attention_mask = attention_mask.cuda()
        outputs = model(input_ids=input_ids, token_type_ids=token_type, attention_mask=attention_mask)
        sim[j] = outputs[0].squeeze(-1)
    rank = np.argsort(-sim)
    for j in rank[:100]:
        f.write(docs[j]+" ")
    f.write("\n")


# # for one input

# In[ ]:


q = "I am fool"
d = "hello how are you , kekkek mekwjsa ,edas,dl"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
cls_token = tokenizer.cls_token
sep_token = tokenizer.sep_token

q = f"{cls_token} {q} {sep_token}"
print(q)

query = tokenizer.tokenize(q)
query_ids = tokenizer.convert_tokens_to_ids(query)
print(query_ids)

doc_len = 512-len(query_ids)
doc = tokenizer.tokenize(d)
doc_ids = tokenizer.convert_tokens_to_ids(doc)
doc_ids = doc_ids[:doc_len]
print(doc_ids)

ids = query_ids + doc_ids
ids = torch.LongTensor([ids])

attention_mask = [*(1 for _ in query_ids), *(1 for _ in doc_ids)]
attention_mask = torch.LongTensor([attention_mask])

labels = 1
labels = torch.FloatTensor([labels])

batch = (ids, attention_mask, labels)


# In[ ]:


bert_finetune = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
optimizer = optim.Adam(bert_finetune.parameters(), lr=2e-5)
optimizer.zero_grad()
criterion = nn.BCEWithLogitsLoss()
bert_finetune.to(device)
bert_finetune.train()
ids = ids.cuda()
attention_mask = attention_mask.cuda()
labels = labels.cuda()

print(ids.shape)
print(attention_mask.shape)
print(labels.shape)


# In[ ]:


_, outputs = bert_finetune(input_ids=ids, attention_mask=attention_mask, labels=labels)
print(_)
print(outputs)
loss = criterion(outputs.squeeze(-1), labels)
loss.backward()
optimizer.step()
optimizer.zero_grad()
print()
print(outputs.squeeze(-1))
print(labels)
print(loss)


# In[ ]:




