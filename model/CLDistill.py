from data_helper import read_catgy_text, load_parl

from keras.optimizers import *
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, Convolution2D, AveragePooling1D
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.models import Model, load_model, model_from_json

from sklearn.preprocessing import normalize
import random
import math

from model.kimCNN_text import kimCNN
from model.adversarial import train_for_adversarial_GRL, get_discriminator
from model.data_helper import accuracy
from model.distillation import simple_distill_parl, restore_weights, rev_word_idx

import os
from os.path import join, isfile

def my_pad_sequences(X, maxlen):
    if X is not None:
        return sequence.pad_sequences(X, maxlen)
    else:
        return X

def my_save_model(model, f):
    # serialize model to JSON
    model_json = model.to_json()
    with open(f + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f + ".h5")
    print("Saved model into h5 file!")

def my_load_model(f):
    json_file = open(f + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f + ".h5")
    print("Loaded model from disk")
    return loaded_model

def get_time_stamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

def idx2str(l, idx):
    inv_idx = {v: k for k, v in idx.iteritems()}
    s = [inv_idx[e] for e in l if e > 0]
    return ' '.join(s)

def write_dict(f, d):
    for k, v in d.iteritems():
        f.write(str(k) + ':' + str(v) + '\n')


class CLD(object):
    def __init__(self, config):
        self.config = config

    def read_src(self, train_path, emb_path):
        self.X_src_train, self.y_src_train,\
        self.src_emb, self.src_nb_words,\
        self.src_maxlen, self.catgy_ind,\
        self.src_word_idx_map = read_catgy_text(train_path, emb_path)
        return 0

    def read_tgt(self, train_path, emb_path):
        self.X_tgt_test, self.y_tgt_test,\
        self.tgt_emb, self.tgt_nb_words,\
        self.tgt_maxlen, _,\
        self.tgt_word_idx_map = read_catgy_text(train_path, emb_path)
        return 0

    def read_parl(self, parl_path):
        self.X_src_parl, self.X_tgt_parl = load_parl(parl_path, self.src_word_idx_map, self.tgt_word_idx_map, self.src_maxlen, self.tgt_maxlen)
        return 0

    def normalize_emb(self):
        # source training data
        self.src_emb = normalize(self.src_emb, axis=1, norm='l2')
        print('Source pretrained embedding size:', self.src_emb.shape)
        print('Source max features:', self.src_nb_words)

        # target training data
        self.tgt_emb = normalize(self.tgt_emb, axis=1, norm='l2')
        print('Target pretrained embedding size:', self.tgt_emb.shape)
        print('Target max features:', self.tgt_nb_words)

    def config_param(self, params): # TODO
        # source and target parameters
        src_params = dict(params)
        tgt_params = dict(params)

        # set feature model-depandant params
        hp = params['nn_hp']
        for k in hp[0]:
            src_params[k] = hp[0][k]
        if len(hp) > 1: # use target specific hp
            for k in hp[1]:
                tgt_params[k] = hp[1][k]
        else:
            for k in hp[0]:
                tgt_params[k] = hp[0][k]
        
        src_params['maxlen'] = self.src_maxlen
        tgt_params['maxlen'] = self.tgt_maxlen
        src_params['embedding_dims'] = self.src_emb.shape[1]
        tgt_params['embedding_dims'] = self.tgt_emb.shape[1]
        tgt_params['max_features'] = self.tgt_emb.shape[0]
        src_params['max_features'] = self.src_emb.shape[0]

        return src_params, tgt_params

    def print_train_info(self):
        print('pretrained source embedding shape:', self.src_emb.shape)
        print('pretrained target embedding shape:', self.tgt_emb.shape)
        print(len(self.X_src_train), 'train sequences')
        print(len(self.X_tgt_test), 'test sequences')

        print('X_tgt_test max:', self.X_tgt_test.max())

        print('X_src_train shape:', self.X_src_train.shape)
        print('y_src_train shape:', self.y_src_train.shape)
        
        print('X_tgt_test shape:', self.X_tgt_test.shape)
        print('X_src_parl shape:', self.X_src_parl.shape)
        print('X_tgt_parl shape:', self.X_tgt_parl.shape)

    def eval(self, out_path=None):
        y_tgt_test_pred = self.tgt_model.end2end_model.predict(self.X_tgt_test, batch_size=self.config['batch_size'])
        print('y_tgt_test_pred[:10]:', y_tgt_test_pred[:10])
        print('self.y_tgt_test[:10]:', self.y_tgt_test[:10])
        acc = accuracy(self.y_tgt_test, y_tgt_test_pred)
        print('Accuraccy: ', acc)
        if out_path is not None:
            out_f = open(out_path, 'w')
            out_f.write(str(acc))

    def train(self):
        # set parameters:
        self.config['label_dims'] = self.y_src_train.shape[1]
        self.normalize_emb()

        src_params, tgt_params = self.config_param(self.config)
        self.print_train_info()

        print('Building model...')
        src_model = kimCNN(src_params)
        tgt_model = kimCNN(tgt_params)

        # build models
        tgt_model.build_model(pretrained_embedding=self.tgt_emb, temperature=self.config['temp'])

        # train the source model
        if not isfile(join(self.config['save_path'], 'src_model.json')):
            src_model.build_model(pretrained_embedding=self.src_emb, temperature=self.config['temp'])
            if self.config['adv'] == False:
                # standard training of source model
                file_path = join(self.config['save_path'], 'tmp_model_' + str(os.getpid()) + '.model')
                checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
                early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
                callbacks_list = [checkpoint, early_stopping]
                src_model.end2end_model.fit(self.X_src_train, self.y_src_train,
                      batch_size=self.config['batch_size'],
                      nb_epoch=self.config['nb_epoch'],
                      validation_split=0.2,
                      callbacks=callbacks_list)
                src_model.end2end_model.load_weights(file_path)
                os.remove(file_path)
            else:
                # adversarial training of source model with parallel text
                adv_hp = self.config['adv_hp']
                domain_discriminator = get_discriminator(src_model.feature_extractor.output_shape[1], hidden_dim=adv_hp['adv_disc_hidden_dim'], depth=adv_hp['adv_disc_depth'], dropout_rate=adv_hp['dropout_rate'])
                train_for_adversarial_GRL(src_model, domain_discriminator, self.X_src_train[:int(math.floor(self.X_src_train.shape[0]*0.8)),:], self.y_src_train[:int(math.floor(self.X_src_train.shape[0]*0.8))], self.X_src_train[int(math.floor(self.X_src_train.shape[0]*0.8)):,:], self.y_src_train[int(math.floor(self.X_src_train.shape[0]*0.8)):], self.X_src_parl, nb_epoch=adv_hp['nb_epoch'], batch_size=adv_hp['batch_size'], lr=adv_hp['lr'], k=adv_hp['k'], l=adv_hp['l'], hp_lambda=adv_hp['hp_lambda'], plt_frq=adv_hp['plt_frq'], plot=[join(self.config['save_path'], 'adv.src_trn.png'), join(self.config['save_path'], 'adv.src_val.png')], use_soft_model=False)

            my_save_model(src_model.end2end_model, join(self.config['save_path'], 'src_model'))
            
        else:
            src_end2end_model = my_load_model(join(self.config['save_path'], 'src_model'))
            src_model.build_model(pretrained_embedding=self.src_emb, temperature=self.config['temp'])
            restore_weights(src_end2end_model, src_model.end2end_soft_model)
            restore_weights(src_end2end_model, src_model.feature_extractor)

        # train the target model with distillation
        simple_distill_parl(src_model, tgt_model, self.config, self.X_src_parl, self.X_tgt_parl, adv=self.config['adv'], adv_hp=self.config['adv_hp'], X_tgt_unlabeled=self.X_tgt_test, X_tgt_val=None, y_tgt_val=None, tgt_idx_word=rev_word_idx(self.tgt_word_idx_map), save_path=self.config['save_path'])

        self.tgt_model = tgt_model
