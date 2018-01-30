import sys, os
import errno 
import json
import numpy as np
import cPickle, shutil
from os import listdir 
from os.path import join
import argparse
from model.CLDistill import CLD
from config import get_train_config

# -----------------------------
# Train a text classifier using using method in "Cross-lingual Distillation for Text Classification"
# -----------------------------

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def main():
    parser = argparse.ArgumentParser(description='Train a text classifier in the target language,\
                                                 using training data in the source language and \
                                                 parallel data between the two languages')
    parser.add_argument('-src_train_path', default='data/amazon_review/en/book/train')
    parser.add_argument('-src_emb_path', default='data/amazon_review/en/all.review.vec.txt')
    parser.add_argument('-tgt_test_path', default='data/amazon_review/de/book/train')
    parser.add_argument('-tgt_emb_path', default='data/amazon_review/de/all.review.vec.txt')
    parser.add_argument('-parl_data_path', default='data/amazon_review/de/book/parl')
    parser.add_argument('-save_path', default='experiments/en-de/book')
    parser.add_argument('-dataset', default='amazon_review')
    args = parser.parse_args()

    # load the configuration file
    config = get_train_config(dataset=args.dataset)

    # save the configuration file
    mkdir_p(args.save_path)
    config['save_path'] = args.save_path
    with open(join(args.save_path, 'config.json'), 'w') as outfile:
        json.dump(config, outfile)

    # initialize a model object
    model = CLD(config)

    # read src training data
    model.read_src(train_path=args.src_train_path, emb_path=args.src_emb_path)
    # read tgt testing data
    model.read_tgt(train_path=args.tgt_test_path, emb_path=args.tgt_emb_path)
    # read parallel data
    model.read_parl(parl_path=args.parl_data_path)
    # let's start the cross-lingual training
    model.train()
    # let's make prediction on test data and evaluate
    model.eval(join(args.save_path, 'acc.txt'))

if __name__ == '__main__':
    main()