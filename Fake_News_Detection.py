# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:01:03 2021

@author: naisa
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#import requests
#import bs4
#from bs4 import BeautifulSoup
import torch.nn as nn
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
#from transformers import AlbertTokenizer, AlbertModel
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
#from flask import Flask, jsonify, request

df_true = pd.read_csv('True.csv')
df_true['Target'] = 1
df_fake = pd.read_csv('Fake.csv')
df_fake['Target'] = 0

data_df = pd.concat([df_true, df_fake]).reset_index(drop = True)
data_df = data_df.sample(frac=1).reset_index(drop=True)

#data_df['Target'].describe()

# Checking GPU availability
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)

X_train, X_fin_test, y_train, y_fin_test = train_test_split(data_df['text'], data_df['Target'], test_size=0.05, random_state=42)
train = pd.DataFrame()
train['text'] = X_train
train['Target'] = y_train 

X_train, X_test, y_train, y_test = train_test_split(train['text'], train['Target'], test_size=0.15, random_state=42)

train_texts = X_train.to_list()
train_labels = y_train.to_list()
test_texts = X_test.to_list()
test_labels = y_test.to_list()

#tokenizing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:255], train_texts))
test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:255], test_texts))
train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
test_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))
train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=256, truncating="post", padding="post", dtype="int")
test_tokens_ids = pad_sequences(test_tokens_ids, maxlen=256, truncating="post", padding="post", dtype="int")

# for masking
train_y = np.array(train_labels) == 1
test_y = np.array(test_labels) == 1

# Generating masks
train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]

# Converting into tensors(Pytorch)
train_masks_tensor = torch.tensor(train_masks)#.to(device)
test_masks_tensor = torch.tensor(test_masks)#.to(device)
train_tokens_tensor = torch.tensor(train_tokens_ids)#.to(device)
train_y_tensor = torch.tensor(train_y.reshape(-1, 1)).float()
#train_y_tensor = train_y_tensor.to(device)
test_tokens_tensor = torch.tensor(test_tokens_ids)#.to(device)
test_y_tensor = torch.tensor(test_y.reshape(-1, 1)).float()
#test_y_tensor = test_y_tensor.to(device)




#model
class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, tokens, masks=None):
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba

# Config setting
BATCH_SIZE = 4
EPOCHS = 3

# Making dataloaders
train_dataset =  torch.utils.data.TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)
train_sampler =  torch.utils.data.RandomSampler(train_dataset)
train_dataloader =  torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)
test_dataset =  torch.utils.data.TensorDataset(test_tokens_tensor, test_masks_tensor, test_y_tensor)
test_sampler =  torch.utils.data.SequentialSampler(test_dataset)
test_dataloader =  torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

bert_clf = BertBinaryClassifier()
bert_clf = bert_clf.cuda()
#wandb.watch(bert_clf)
optimizer = torch.optim.Adam(bert_clf.parameters(), lr=3e-6)

# training 
for epoch_num in range(EPOCHS):
    bert_clf.train()
    train_loss = 0
    for step_num, batch_data in enumerate(train_dataloader):
        token_ids, masks, labels = tuple(t for t in batch_data)
        token_ids, masks, labels = token_ids.to(device), masks.to(device), labels.to(device)
        token_ids = token_ids.long()
        probas = bert_clf(token_ids, masks)
        loss_func = nn.BCELoss()
        batch_loss = loss_func(probas, labels)
        train_loss += batch_loss.item()
        bert_clf.zero_grad()
        batch_loss.backward()
        optimizer.step()
#        wandb.log({"Training loss": train_loss})
#        print('Epoch: ', epoch_num + 1 )
        if step_num % 100 == 0:
            print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(X_train) / BATCH_SIZE, train_loss / (step_num + 1)))
#            print(epoch_num, step_num, train_loss)
########################## test ##############################################################################################
        
bert_clf.eval()
bert_predicted = []
all_logits = []
with torch.no_grad():
    test_loss = 0
    for step_num, batch_data in enumerate(test_dataloader):
        token_ids, masks, labels = tuple(t for t in batch_data)
        token_ids, masks, labels = token_ids.to(device), masks.to(device), labels.to(device)
        token_ids = token_ids.long()
        logits = bert_clf(token_ids, masks)
        loss_func = nn.BCELoss()
        loss = loss_func(logits, labels)
        test_loss += loss.item()
        numpy_logits = logits.cpu().detach().numpy()
        bert_predicted += list(numpy_logits[:, 0] > 0.5)
        all_logits += list(numpy_logits[:, 0])
        

#saving weights
torch.save(bert_clf.state_dict(), 'nb_state257.pth')                                                                                     
