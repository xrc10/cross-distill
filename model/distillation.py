import sys
import os
from keras.models import Model
from keras.layers import Input
from keras.callbacks import ReduceLROnPlateau
from adversarial import *
import math
import numpy as np
from data_helper import *
from keras.callbacks import ModelCheckpoint, EarlyStopping

def simple_distill_parl(src_model, tgt_model, params, X_src_parl, X_tgt_parl, X_tgt_val=None, y_tgt_val=None, adv=False, adv_hp=None, X_tgt_unlabeled=None, X_tgt_trn=None, y_tgt_trn=None, tgt_idx_word=None, save_path=None):
    """ distillation method """
    file_path = os.path.join(params['save_path'], 'distill_parl_' + str(os.getpid()) + '.model') 
    soft_pred = src_model.end2end_soft_model.predict(X_src_parl, batch_size=params['batch_size'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                  patience=0, min_lr=0.00001)
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
    callbacks_list = [checkpoint, early_stopping, reduce_lr]
    # if X_tgt_trn is not None and y_tgt_trn is not None:
    #   X_tgt_parl = np.concatenate((X_tgt_parl, X_tgt_trn), axis=0)
    #   soft_pred = np.concatenate((soft_pred, y_tgt_trn), axis=0)
    if adv == False:
      if X_tgt_val is None or y_tgt_val is None:
        tgt_model.end2end_soft_model.fit(X_tgt_parl, soft_pred,
              batch_size=params['batch_size'],
              nb_epoch=params['nb_epoch'],
              # validation_split=0.2,
              # callbacks=callbacks_list,
              )

      else:
        tgt_model.end2end_soft_model.fit(X_tgt_parl, soft_pred,
              batch_size=params['batch_size'],
              nb_epoch=params['nb_epoch'],
              validation_data=(X_tgt_val, y_tgt_val),
              callbacks=callbacks_list)
      # tgt_model.end2end_soft_model.load_weights(file_path)
      # os.remove(file_path)
    else:
      val_metric='loss'
      domain_discriminator = get_discriminator(tgt_model.feature_extractor.output_shape[1], hidden_dim=adv_hp['adv_disc_hidden_dim'], depth=adv_hp['adv_disc_depth'], dropout_rate=adv_hp['dropout_rate'])
      if X_tgt_val is None or y_tgt_val is None:
        # X_trn = X_tgt_parl[:int(math.floor(X_tgt_parl.shape[0]*0.8)),:]
        # y_trn = soft_pred[:int(math.floor(X_tgt_parl.shape[0]*0.8))]
        # X_val = X_tgt_parl[int(math.floor(X_tgt_parl.shape[0]*0.8)):,:]
        # y_val = soft_pred[int(math.floor(X_tgt_parl.shape[0]*0.8)):]
        X_trn = X_tgt_parl
        y_trn = soft_pred
        X_val = None
        y_val = None
      else:
        X_trn = X_tgt_parl
        y_trn = soft_pred
        X_val = X_tgt_val
        y_val = y_tgt_val
        val_metric = 'acc'
      train_for_adversarial_GRL(tgt_model, domain_discriminator, X_trn, y_trn, X_val, y_val, X_tgt_unlabeled, nb_epoch=adv_hp['nb_epoch'], batch_size=adv_hp['batch_size'], lr=adv_hp['lr'], k=adv_hp['k'], l=adv_hp['l'], hp_lambda=adv_hp['hp_lambda'], plt_frq=adv_hp['plt_frq'], plot=[join(params['save_path'],'adv.tgt_parl.png'), join(params['save_path'], 'adv.tgt_val.png')], use_soft_model=True, val_metric=val_metric)

    # if X_tgt_trn is not None and y_tgt_trn is not None and X_tgt_trn.shape[0]>0:
    #   if X_tgt_val is None or y_tgt_val is None:
    #     tgt_model.end2end_model.fit(X_tgt_trn, y_tgt_trn,
    #           batch_size=params['batch_size'],
    #           nb_epoch=params['nb_epoch'],
    #           validation_split=0.2,
    #           callbacks=callbacks_list)
    #   else:
    #     tgt_model.end2end_model.fit(X_tgt_trn, y_tgt_trn,
    #           batch_size=params['batch_size'],
    #           nb_epoch=params['nb_epoch'],
    #           validation_data=(X_tgt_val, y_tgt_val),
    #           callbacks=callbacks_list)

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