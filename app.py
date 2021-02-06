# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 19:39:16 2021

@author: naisa
"""

from pytorch_pretrained_bert import BertTokenizer, BertModel
import os

from keras.preprocessing.sequence import pad_sequences
import torch.nn as nn
import torch 
from flask import Flask, request, make_response
device = torch.device('cpu')

fakeApp = Flask(__name__)

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


bert_clf = BertBinaryClassifier().to(device)

bert_clf.load_state_dict(torch.load('nb_state257.pth'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def predict(article):
#    bert_predicted = []
#    all_logits = []
    global bert_clf, tokenizer, device
    test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:255], [article]))
    test_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))
    test_tokens_ids = pad_sequences(test_tokens_ids, maxlen=256, truncating="post", padding="post", dtype="int")
    test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]
    test_masks_tensor = torch.tensor(test_masks).to(device)
    test_tokens_ids = torch.tensor(test_tokens_ids).to(device).long()
    with torch.no_grad():
        logits = bert_clf(test_tokens_ids, test_masks_tensor)
        numpy_logits = logits.cpu().detach().numpy()
    return numpy_logits[0, 0]


@fakeApp.route('/predict', methods=['POST'])
def get_prediction():
    data = request.get_json()
    article = data['article']
    prob = predict(article)
    
    return make_response({'prob': str(prob)}, 200)
    

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    
    fakeApp.run(debug=False, host = '0.0.0.0', port = port)