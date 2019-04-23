
# coding: utf-8

# In[1]:


import argparse
import time
import random
import glob
import multiprocessing as mp

import numpy as np

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import HybridBlock
from mxnet.gluon.data import DataLoader

import gluonnlp as nlp


import d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn

import os
import csv
import multiprocessing as mp
from gluonnlp import Vocab, data
from mxnet.gluon.data import ArrayDataset, SimpleDataset
from nltk import word_tokenize
import pandas as pd
import re


# In[13]:


class QuoraDataset(ArrayDataset):
    """This dataset provides access to Quora insincere data competition"""

    def __init__(self, segment, root_dir="../input/"):
        self._root_dir = root_dir
        self._segment = segment
        self._segments = {
            # We may change the file path
            'train': '/Applications/files/classes_homework/Berkeley_ieor/STAT157/project/train.csv',
            'test': '/Applications/files/classes_homework/Berkeley_ieor/STAT157/project/test.csv'
        }

        super(QuoraDataset, self).__init__(self._read_data())

    def _read_data(self):
        file_path = os.path.join(self._root_dir, self._segments[self._segment])
        with open(file_path, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            # ignore 1st line - which is header
            data = [tuple(row) for i, row in enumerate(reader) if i > 0]
            for i in range(len(data)):
                data[i] = data[i][1:3]
                data[i] = list(data[i])
                data[i][1] = int(data[i][1])
        return data
    
    
    


# In[15]:


train_dataset = QuoraDataset('train')


# In[23]:


def preprocess_quora(data, vocab):  
    max_l = 200  

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = d2l.get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels



# In[25]:


batch_size = 64
train_data, test_data = nlp.data.train_valid_split(train_dataset,valid_ratio = 0.3)
vocab = d2l.get_vocab_imdb(train_data)
train_iter = gdata.DataLoader(gdata.ArrayDataset(
    *preprocess_quora(train_data, vocab)), batch_size, shuffle=True)
test_iter = gdata.DataLoader(gdata.ArrayDataset(
    *preprocess_quora(test_data, vocab)), batch_size)




# In[30]:


for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
'#batches:', len(train_iter)


# In[26]:


class TextCNN(nn.Block):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
       
        self.pool = nn.GlobalMaxPool1D()
        self.convs = nn.Sequential()  
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))
    
    def forward(self, inputs):
        
        embeddings = nd.concat(
            self.embedding(inputs), self.constant_embedding(inputs), dim=2)
        
        embeddings = embeddings.transpose((0, 2, 1))
        
        encoding = nd.concat(*[nd.flatten(
            self.pool(conv(embeddings))) for conv in self.convs], dim=1)
        
        outputs = self.decoder(self.dropout(encoding))
        return outputs


# In[27]:


embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
ctx = d2l.try_all_gpus()
net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=ctx)


# In[28]:


glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.collect_params().setattr('grad_req', 'null')


# ### Train

# In[ ]:


lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)


# ### Predict

# In[2]:


def predict_textCNN(sentence):
    sentence_ = re.sub("[^a-zA-Z]", " ",sentence)
    sentence_new = sentence_.split()
    sentence_new = nd.array(vocab.to_indices(sentence_new), ctx=d2l.try_gpu())
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return '1' if label.asscalar() == 1 else '0'

