from keras.optimizers import *
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Merge, TimeDistributed
from keras.layers import Embedding
from keras.layers import Input
from GradientReversalLayer import GradientReversalLayer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
import random
from tqdm import tqdm
from keras.models import model_from_json
import datetime
import time
import os
import sys

def restore_weights(src_model, tgt_model, verbose=0):
    src_ws = src_model.get_weights()
    tgt_ws = tgt_model.get_weights()
    if verbose > 0:
      print 'source model size:'
      for w in src_ws:
        print w.shape
        if w.shape[0] < 10 and len(w.shape) == 1:
          print w
        print '------'

    tgt_model.set_weights(src_ws[:len(tgt_ws)])
    tgt_ws = tgt_model.get_weights()
    if verbose > 0:
      print 'target model size:'
      for w in tgt_ws:
        print w.shape
        if w.shape[0] < 10 and len(w.shape) == 1:
          print w
        print '------'
    return 0

def get_time_stamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

def get_discriminator(input_dim, hidden_dim=100, depth=2, dropout_rate=0.2):
    feature_input = Input(shape=(input_dim, ), dtype='float32')
    x = Dense(hidden_dim)(feature_input)
    x = Activation('relu')(x)
    for _ in range(depth-1):
        x = Dense(hidden_dim)(x)
        x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(2)(x)
    x = Activation('softmax')(x)
    discriminator = Model(input=feature_input, output=x)

    return discriminator

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def make_trainable_by_index(net, val, index):
    for i,l in enumerate(net.layers):
        if i in index:
            l.trainable = val

def plot_loss(losses, fig_file):
#        display.clear_output(wait=True)
#        display.display(plt.gcf())
    plt.figure(figsize=(10,8))
    plt.plot(losses["topic"], label='topic classification loss')
    if "domain" in losses:
        plt.plot(losses["domain"], label='domain classification loss')
    plt.legend()
    plt.savefig(fig_file)

def get_mini_batch_idx(size, mini_batch_size):
    idx = np.arange(size)
    random.shuffle(idx)
    mini_batches = [
        idx[k:min(k+mini_batch_size,size)]
        for k in range(0, size, mini_batch_size)
        ]
    return mini_batches

def train_for_adversarial_GRL(model, domain_discriminator, X_src_train, y_src_train, X_tgt_val, y_tgt_val, X_tgt_unlabeled, nb_epoch=5000, batch_size=32, lr=1e-4, k=1, l=1, hp_lambda=1e-7, plt_frq=1, plot=['adv_trn_loss.png', 'adv_val_loss.png'], use_soft_model=True, val_patience_count = 3, val_metric='loss'):
    # use the Gradient Reversal Layer of training adversarial network

    if use_soft_model:
        end2end_model = model.end2end_soft_model
        end2end_model_opt = model.end2end_soft_opt
    else:
        end2end_model = model.end2end_model
        end2end_model_opt = model.end2end_opt

    # set up loss storage vector
    src_losses = {"topic":[], "domain":[]}
    tgt_losses = {"topic":[]}
    val_no_imprv_count = 0
    minium_count = 0
    domain_discriminator_opt = Adam(lr=lr)

    time_stamp = get_time_stamp()
    weights_file_name = '/tmp/tmp_model.' + time_stamp + str(os.getpid()) + '.h5'
    json_file_name = '/tmp/tmp_model.' + time_stamp + str(os.getpid()) + '.json'

    domain_discriminator_input = Input(shape=(model.maxlen, ), dtype='float32')
    x = model.feature_extractor(domain_discriminator_input)
    x = GradientReversalLayer(hp_lambda=hp_lambda)(x)
    x = domain_discriminator(x)
    domain_discriminator_model = Model(input=domain_discriminator_input, output=x)
    domain_discriminator_model.compile(loss='categorical_crossentropy',
                      optimizer=domain_discriminator_opt,
                      metrics=['categorical_accuracy'])
    
    val_loss = float("inf")
    val_acc = 0.0
    prev_obj = None

    for e in range(nb_epoch):
        print('Iteration: ' + str(e))
        cur_obj = 0.0
        mini_batches = get_mini_batch_idx(X_src_train.shape[0], batch_size)
        # for batch_idx, src_batch_idx in enumerate(tqdm(mini_batches)):
        for batch_idx, src_batch_idx in enumerate(mini_batches):
            # src_batch_idx = np.random.choice(X_src_train.shape[0], size=batch_size, replace=False)
            # print(src_batch_idx.shape)
            X_src_train_batch = X_src_train[src_batch_idx,:]
            y_src_train_batch = y_src_train[src_batch_idx]

            # train domain discriminator
            tgt_batch_idx = np.random.choice(X_tgt_unlabeled.shape[0], size=X_src_train_batch.shape[0], replace=False)
            X_tgt_unlabeled_batch = X_tgt_unlabeled[tgt_batch_idx,:]

            # update discriminator and feature extractor at same time
            y = np.zeros([2*X_src_train_batch.shape[0],2])
            y[:X_src_train_batch.shape[0],1] = 1
            y[X_src_train_batch.shape[0]:,0] = 1
            X_comb = np.concatenate((X_src_train_batch, X_tgt_unlabeled_batch))
            for inner in range(k):
                # if inner < k-1:
                #     make_trainable_by_index(domain_discriminator_model, False, [1])
                # else:
                #     make_trainable_by_index(domain_discriminator_model, False, [1])
                # update feature extractor and domain discriminator
                loss = domain_discriminator_model.train_on_batch(X_comb, y)
                end2end_domain_loss = loss[0]
                src_losses['domain'].append(end2end_domain_loss)

            # train source domain topic classifier
            # make_trainable_by_index(domain_discriminator_model, True, [1])
            for _ in range(l):
                end2end_topic_loss = end2end_model.train_on_batch(X_src_train_batch, y_src_train_batch)
            src_losses['topic'].append(end2end_topic_loss[0])


            if plot is not None and (batch_idx+1)%plt_frq == 0:
                plot_loss(src_losses, plot[0])

        cur_obj = end2end_topic_loss[0]

        # validation
        if X_tgt_val is not None and e >= minium_count:
            # validation
            tgt_loss = end2end_model.evaluate(X_tgt_val, y_tgt_val, batch_size=batch_size, verbose=0)
            tgt_losses["topic"].append(tgt_loss[0])
            print("\n\ntopic val loss: " + str(tgt_loss[0]) + ", accuracy:" + str(tgt_loss[1]))

            if plot:
                plot_loss(tgt_losses, plot[1])

            if (val_metric == 'loss' and tgt_loss[0] < val_loss) or (val_metric == 'acc' and tgt_loss[1] > val_acc):
                val_no_imprv_count = 0
                val_loss = min(tgt_loss[0], val_loss)
                val_acc = max(tgt_loss[1], val_acc)
                end2end_model.save_weights(weights_file_name)
                model_json = end2end_model.to_json()
                with open(json_file_name, "w") as json_file:
                    json_file.write(model_json)
            else:
                val_no_imprv_count += 1

            if val_no_imprv_count >= val_patience_count:
                print('early stop at epoch:(by validation performance)', e)
                break # early stop based on validation data

        elif e >= minium_count:
            if prev_obj is None:
                prev_obj = cur_obj
            else:
                print('prev_obj:', prev_obj)
                if abs(prev_obj-cur_obj)/abs(prev_obj) < 1e-5:
                    print('early stop at epoch:(by convergence)', e)
                    break
                else:
                    prev_obj = cur_obj

        plt.close('all')
    # get the best model
    # end2end_model = load_model('model/tmp_model.h5')
    if X_tgt_val is not None:
        # load json and create model
        json_file = open(json_file_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        end2end_model = model_from_json(loaded_model_json)
        # load weights into new model
        end2end_model.load_weights(weights_file_name)
        os.remove(weights_file_name)
        os.remove(json_file_name)

    restore_weights(end2end_model, model.end2end_soft_model)
    restore_weights(end2end_model, model.feature_extractor)
    restore_weights(end2end_model, model.end2end_model)
