'''
import necessary package
'''

import time
import random
import glob

import numpy as np

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import HybridBlock
from mxnet.gluon.data import DataLoader, ArrayDataset

import gluonnlp as nlp


import d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn

import os
import csv
import multiprocessing as mp
from gluonnlp import Vocab, data
from nltk import word_tokenize
import pandas as pd
import re

ctx = d2l.try_gpu()

'''
The following is the function for textCNN
'''
import mxnet as mx
from mxnet import gluon, autograd
ctx = d2l.try_gpu()
from mxnet.gluon import data as gdata, loss as gloss, nn


'''
The following is the function for textCNN
'''

from mxnet.gluon.data import DataLoader
import gluonnlp as nlp
from mxnet import gluon, init, nd
from mxnet.gluon.data import ArrayDataset

#====================the following is creating vocab from train.csv, test.csv=======================
class QuoraDataset(ArrayDataset):
    """This dataset provides access to Quora insincere data competition"""

    def __init__(self, segment, root_dir=""):
        self._root_dir = root_dir
        self._segment = segment
        self._segments = {
            # We may change the file path
            'train': 'train.csv',
            'test': 'test.csv'
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

dataset = QuoraDataset('train')
def trans_data(dataset):
    train_num = int(0.7*len(dataset))
    data, label = [], []
    for pair in dataset:
        data.append(pair[0])
        label.append(pair[1])
    train_data = data[:train_num], label[:train_num]
    test_data = data[train_num:], label[train_num:]
    return train_data, test_data

train_data, test_data = trans_data(dataset)

def tokenize(sentences):
    return [line.split(' ') for line in sentences]

train_tokens = tokenize(train_data[0])
test_tokens = tokenize(test_data[0])

vocab = d2l.Vocab([tk for line in train_tokens for tk in line], min_freq=5)

#===============================================================================================================

class TextCNN(nn.Block):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer does not participate in training
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # The max-over-time pooling layer has no weight, so it can share an
        # instance
        self.pool = nn.GlobalMaxPool1D()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # Concatenate the output of two embedding layers with shape of
        # (batch size, number of words, word vector dimension) by word vector
        embeddings = nd.concat(
            self.embedding(inputs), self.constant_embedding(inputs), dim=2)
        # According to the input format required by Conv1D, the word vector
        # dimension, that is, the channel dimension of the one-dimensional
        # convolutional layer, is transformed into the previous dimension
        embeddings = embeddings.transpose((0, 2, 1))
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, an NDArray with the shape of (batch size, channel size, 1)
        # can be obtained. Use the flatten function to remove the last
        # dimension and then concatenate on the channel dimension
        encoding = nd.concat(*[nd.flatten(
            self.pool(conv(embeddings))) for conv in self.convs], dim=1)
        # After applying the dropout method, use a fully connected layer to
        # obtain the output
        outputs = self.decoder(self.dropout(encoding))
        #print(outputs)
        return outputs

# initialize the net
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
# FIXME: the model size is hard coded
text_cnn_net = TextCNN(68688, embed_size, kernel_sizes, nums_channels)
text_cnn_net.load_parameters('textCNN.params', ctx=ctx)

def predict_text_cnn(sentence):
    if len(sentence) < 5:
        sentence += (5 - len(sentence)) * ' <unk>'
    sentence = nd.array(vocab[sentence.split()], ctx=ctx)
    label = nd.argmax(text_cnn_net(sentence.reshape((1, -1))), axis=1)
    return label.asscalar()

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
    return out[0].argmax(axis=0).asscalar()
