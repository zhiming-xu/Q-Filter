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

ctx = d2l.try_gpu()

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


def predict_cnn(sentence):
    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    vocab = d2l.get_vocab_imdb(sentence)
    net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)

    net.load_parameters('textCNN.params', ctx=ctx)

    sentence_ = re.sub("[^a-zA-Z]", " ", sentence)
    sentence_new = sentence_.split()
    sentence_new = nd.array(vocab.to_indices(sentence_new), ctx=ctx)
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

# this function will take sentences (a tuple of strings) and other
# supporting data as input, transform the sentences into data that
# can be fed into bert model for prediction

def bert_transform(sentence, tokenizer, max_seq_length, \
                   pad=True, pair=True):
    bert_trans = BERTSentenceTransform(tokenizer, max_seq_length,\
                                       pad=pad, pair=pair)
    input_ids, valid_length, segment_ids = bert_trans(sentence)

    return input_ids, valid_length, segment_ids

# this is the interface to interact with bert model, take a tuple of
# sentences (strings) as input, this function should return the prediction
# for each of them
# FIXME: FOR NOW, IT CAN ONLY TAKE IN A LIST, WHOSE ELEMENT IS `ONE`
# STRING, i.e. ['some string in this field'], AND RETURN 1 IFF INSINCERE

# this part is from original finetune_classfier file
bert, vocabulary = nlp.model.get_bert_model(model_name='bert_12_768_12', \
                   dataset_name='book_corpus_wiki_en_uncased', \
                   pretrained=True, ctx=ctx, use_pooler=True, \
                   use_decoder=False, use_classifier=False)

model = BERTClassifier(\
        bert, dropout=.1, num_classes=len(QuoraDataset.get_labels())
        )
model.load_parameters('model_bert_Quora_3.params', ctx=ctx)
model.hybridize(static_alloc=True)
 
# from here in finetune_classfier/preprocess_data 
tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)
max_seq_length = 32

# this function takes in exactly one sentence and predict for it,
# note that the input should be a list containing only one string
# i.e. ['question we want to predict']. it returns 1 iff. insincere,
# returns 0 otherwise
def predict_bert(sentence):
 
    # [FIXED] this function's ouput is wrong, it does not return word embedding
    # the bug is fixed, we need to set the input size to be batch_size * valid_length
    # i.e. (8, 32) to make this model work because it's trained this way
    inputs_ids, valid_length, type_ids = \
        bert_transform(sentence, tokenizer, max_seq_length, \
                       pad=True, pair=False)

    # since for now we only take one sentence, we need to repeat it for 8 times,
    # each time produces exactly same result
    inputs_ids = nd.repeat(nd.array(inputs_ids).reshape(1, -1), repeats=8, axis=0)
    valid_length = nd.ones(8) * int(valid_length)
    type_ids = nd.repeat(nd.array(type_ids).reshape(1, -1), repeats=8, axis=0)
    out = model(inputs_ids.as_in_context(ctx).astype('int32'), \
                type_ids.as_in_context(ctx).astype('int32'), \
                valid_length.as_in_context(ctx).astype('float32'))
    # the out array is in the form of batch_size * num_classes, i.e. (8, 2)
    # since each row is the same, we only extract the first row and do argmax
    return out[0].argmax(axis=0)
