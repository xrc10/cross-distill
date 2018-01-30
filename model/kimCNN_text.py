'''
This is a keras implementation of simple Convolutional Neural Networks for Sentence Classification. Modified from keras examples at https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import cPickle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.layers import Embedding, Lambda
from keras.layers import Convolution1D, MaxPooling1D, Convolution2D, AveragePooling1D
from keras.layers import Input, TimeDistributed
from keras.datasets import imdb
from keras import backend as K
from keras.models import Model
from keras.constraints import maxnorm
from keras.optimizers import *
from sklearn.metrics import f1_score, accuracy_score
import sys 
import os
sys.stderr = open(sys.argv[0]+'_errorlog.txt', 'w')
import argparse

# import tensorflow as tf
# from keras import backend as K
# with K.tf.device('/gpu:3'):

#     gpu_options = tf.GPUOptions(allow_growth =True)
#     K.set_session(K.tf.Session(config=K.tf.ConfigProto(gpu_options=gpu_options)))

def output_list_of_layer(layer_list, input_layer):
    output_layer_list = list()
    for layer in layer_list:
        output_layer_list.append(layer(input_layer))
    return output_layer_list

class kimCNN:
    def __init__(self, params):
        self.max_features = params['max_features']
        self.batch_size = params['batch_size']
        self.nb_filter = params['nb_filter']
        self.filter_length_list = params['filter_length_list']
        self.nb_epoch = params['nb_epoch']
        self.dropout_rate = params['dropout_rate']
        self.use_pretrained_embedding = params['use_pretrained_embedding']
        self.l2_constraint = params['l2_constraint']
        # parameters with defaults
        if 'pool' in params:
            self.pool = params['pool']
        else:
            self.pool = 'max'

        if 'fix_embedding' in params:
            self.fix_embedding = params['fix_embedding']
        else:
            self.fix_embedding = False

        if 'hidden_dim' in params:
            self.hidden_dim = params['hidden_dim']
        else:
            self.hidden_dim = None

        self.label_type = params['label_type']
        self.embedding_dims = params['embedding_dims']
        self.label_dims = params['label_dims']
        self.maxlen = params['maxlen']
        self.lr = params['lr']

        self.feature_extractor = None
        self.top_logistic_regression = None
        self.end2end_model = None
        self.embedding_lookup = None
        self.embedding_weight = None

    def build_model(self, pretrained_embedding=None, fix_embedding=None, temperature=None):
        if fix_embedding is not None:
            print('Warning: fix_embedding for build_model is deprecated')
        
        fix_embedding = self.fix_embedding
        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        word_input = Input(shape=(self.maxlen, ), dtype='int32', name='word_input')
        self.word_input = word_input
        # add embedding
        if self.use_pretrained_embedding and not pretrained_embedding is None:
        #    pretrained_embedding = np.random.rand(max_features, embedding_dims)
            if fix_embedding:
                self.embedding_lookup = Embedding(output_dim=self.embedding_dims, input_dim=self.max_features, input_length=self.maxlen, weights=[pretrained_embedding], trainable=False)
                embedding_layer = self.embedding_lookup(word_input)
                self.embedding_weight = pretrained_embedding
            else:
                self.embedding_lookup = Embedding(output_dim=self.embedding_dims, input_dim=self.max_features, input_length=self.maxlen, weights=[pretrained_embedding])
                embedding_layer = self.embedding_lookup(word_input)
        else:
            if fix_embedding:
                print('ERROR:Using random embedding as fix!')
                sys.exit(-1)
            self.embedding_lookup = Embedding(output_dim=self.embedding_dims, input_dim=self.max_features, input_length=self.maxlen)
            embedding_layer = self.embedding_lookup(word_input)

        # we add a Convolution1D, which will learn nb_filter
        # word group filters of size filter_length, note here
        # we have more than one filter_length:
        reshaped_embedding_layer = Reshape((self.maxlen, self.embedding_dims))(embedding_layer)

        conv_layer_list = list()
        conv_layer_output_list = list()
        for filter_length in self.filter_length_list:
            conv_layer = Convolution1D(nb_filter=self.nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            W_constraint=maxnorm(self.l2_constraint),
                            subsample_length=1)
            conv_layer_list.append(conv_layer)
            conv_layer_output_list.append((conv_layer(reshaped_embedding_layer)))
                                  
        # we use max pooling for each conv layer:
        pool_layer_output_list = list()
        for i, conv_layer_output in enumerate(conv_layer_output_list):
            if self.pool == 'avg':
                pool_layer = AveragePooling1D(pool_length=self.maxlen+1-self.filter_length_list[i])
            else:
                pool_layer = MaxPooling1D(pool_length=self.maxlen+1-self.filter_length_list[i])
            pool_layer_output_list.append(pool_layer(conv_layer_output))
            

        # We flatten the output of the conv layer,
        # so that we can add a vanilla dense layer:
        flat_layer_output_list = list()
        for i, pool_layer_output in enumerate(pool_layer_output_list):
            flat_layer_output_list.append(Flatten()(pool_layer_output))

        merged_layer_output = Merge(mode='concat', concat_axis=1, name='feature_output')(flat_layer_output_list)

        self.feature_output = merged_layer_output
        # define feature_extractor part from above
        self.feature_extractor = Model(input=word_input, output=merged_layer_output)
        
        if self.hidden_dim is not None:
            merged_layer_output = Dropout(self.dropout_rate)(merged_layer_output)
            merged_layer_output = Dense(self.hidden_dim, activation='tanh')(merged_layer_output)

        # standard logistic regression part
        # feature_input = Input(shape=(self.feature_extractor.output_shape[1], ), dtype='float32')
        x = Dropout(self.dropout_rate)(merged_layer_output)

        # We project onto a single unit output layer, and squash it with a sigmoid:
        logits = Dense(self.label_dims)(x)

        if self.label_type == 'multi-class':
            end2end_output = Activation('softmax')(logits)
        elif self.label_type == 'multi-label':
            end2end_output = Activation('sigmoid')(logits)
        else:
            print('undefined label type {0}'.format(self.label_type))
            sys.exit()

        # self.top_logistic_regression = Model(input=feature_input, output=x)

        # define the end-to-end model
        self.end2end_model = Model(input=word_input, output=end2end_output)
        
        if temperature is not None:
            hot_logits = Lambda(lambda x: x / temperature)(logits)
            if self.label_type == 'multi-class':
                end2end_soft_output = Activation('softmax', name='soft_output')(hot_logits)
            elif self.label_type == 'multi-label':
                end2end_soft_output = Activation('sigmoid', name='soft_output')(hot_logits)
            else:
                print('undefined label type {0}'.format(self.label_type))
                sys.exit()
            # self.top_soft_logistic_regression = Model(input=feature_input, output=soft_output)
            self.soft_output = end2end_soft_output
            self.end2end_soft_model = Model(input=word_input, output=end2end_soft_output)

        if self.label_type == 'multi-class':
            loss_type = 'categorical_crossentropy'
        elif self.label_type == 'multi-label':
            loss_type = 'binary_crossentropy'

        self.end2end_opt = Adam(lr=self.lr)
        self.end2end_model.compile(loss=loss_type,
                      optimizer=self.end2end_opt,
                      metrics=['accuracy', 'categorical_accuracy'])
        if temperature is not None:
            self.end2end_soft_opt = Adam(lr=self.lr)
            self.end2end_soft_model.compile(loss=loss_type,
                          optimizer=self.end2end_soft_opt,
                          metrics=['accuracy', 'categorical_accuracy'])

sys.stderr.close()
sys.stderr = sys.__stderr__