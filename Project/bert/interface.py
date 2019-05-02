'''
import necessary package
'''

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

'''
The following is the function for textCNN
'''
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


def testCNN(sentence):
    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    vocab = d2l.get_vocab_imdb(sentence)
    net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)

    net.load_parameters('textCNN.params', ctx=mx.cpu())

    sentence_ = re.sub("[^a-zA-Z]", " ", sentence)
    sentence_new = sentence_.split()
    sentence_new = nd.array(vocab.to_indices(sentence_new), ctx=mx.cpu())
    label = nd.argmax(net(sentence_new.reshape((1, -1))), axis=1)
    return 'Bad guy detected!! Report to us?' if label.asscalar() == 1 else 'Not bad :)'


'''
The following is the function for BERT
'''
from mxnet import ndarray as nd
from gluonnlp.model import get_bert_model
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform
from bert import BERTClassifier
from dataset import QuoraDataset

ctx = mx.cpu()

# this function will take sentences (a tuple of strings) and other
# supporting data as input, transform the sentences into data that
# can be fed into bert model for prediction
def bert_transform(sentences, tokenizer, max_seq_length, \
                   pad=True, pair=True):
    bert_trans = BERTSentenceTransform(tokenizer, max_seq_length,\
                                       pad=pad, pair=pair)
    input_ids, valid_length, segment_ids = bert_trans(sentences[:-1])

    return input_ids, valid_length, segment_ids

# this is the interface to interact with bert model, take a tuple of
# sentences (strings) as input, this function should return the prediction
# for each of them
def predict_bert(sentences):
    bert, vocabulary = nlp.model.get_model('bert_12_768_12',
                                           dataset_name='book_corpus_wiki_en_uncased',
                                           pretrained=False,
                                           ctx=ctx,
                                           use_pooler=False,
                                           use_decoder=False,
                                           use_classifier=False)

    model = BERTClassifier(\
            bert, dropout=.1, num_classes=len(QuoraDataset.get_labels())
            )
    model.hybridize(static_alloc=True)
    model.load_parameters('./model_bert_Quora_3.params', ctx=ctx, ignore_extra=True)
    bert.cast('float32')

    tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)
    max_seq_length = max([len(sentence) for sentence in sentences])

    inputs_ids, valid_length, type_ids = \
        bert_transform(sentences, tokenizer, max_seq_length, \
                       pad=False, pair=False)
    inputs_ids = nd.array(inputs_ids)
    valid_length = nd.array(valid_length)
    type_ids = nd.array(type_ids)
    out = model(inputs_ids.as_in_context(ctx), type_ids.as_in_context(ctx),\
                valid_length.as_in_context(ctx))
    print(out)
    return out

predict_bert(('this is a sentence', 'this is also a sentence'))
