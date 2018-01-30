""" A data helper to process data """
from os import listdir
from os.path import join, isdir
import sys
import xml.etree.ElementTree as ET
import cPickle
from tqdm import tqdm
import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
import pandas as pd

reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')

def accuracy(y_gold, y_pred):
    # takes a binary <y_gold> and continuous <y_pred>
    # assuming a multi-class classification setting
    max_idx = np.argmax(y_pred, axis=1)
    y_pred = np.zeros(y_pred.shape)
    y_pred[np.arange(y_pred.shape[0]), max_idx] = 1
    l = y_gold.shape[0]
    diff = y_gold - y_pred
    diff = np.sum(np.absolute(diff), axis=1)
    diff = np.equal(diff, np.zeros(diff.shape))
    return float(np.sum(diff))/float(l)

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def add_zero_to_mat(W):
    if W is None:
        return None
    z = np.zeros((1, W.shape[1]), dtype='float32')
    W_new = np.append(z, W, axis=0)
    return W_new

def add_one_to_X(X):
    if X is None:
        return None
    X_new = list()
    for x in X:
        x_new = [t + 1 for t in x]
        X_new.append(x_new)
    return X_new

def load_segments_from_XML(f):
    seg_tok_dict = dict()
    try:
        tree = ET.parse(f)
    except ET.ParseError:
        print 'Oops!  That was wrong XML format!'
        return(-1)
    root = tree.getroot()
    doc = root.find('DOC')
    text = doc.find('TEXT')
    for seg in text.findall('SEG'):
        segId = seg.get('id')
        token_list = list()
        for token in seg.findall('TOKEN'):
            token_list.append(token.text)
        seg_tok_dict[segId] = token_list

    return seg_tok_dict

def toks2idx(toks, idx, maxlen):
    o = list()
    for t in toks[:maxlen]:
        if t in idx:
            o.append(idx[t])
        else:
            # print t.encode('utf-8')
            o.append(0)
    return o

def rev_word_idx(idx):
    rev_idx = {v:k for k ,v in idx.iteritems()}
    rev_idx[0] = 'UNK'
    return rev_idx

def pprint_data(Xs, ys, idx):
    for x,y in zip(list(Xs), list(ys)):
        print y, ' '.join([idx[i] for i in list(x)]).encode('utf-8')

def load_parl(path, src_word_idx, tgt_word_idx, src_maxlen, tgt_maxlen, reverse=False, tgt_char_model=False, seg_char= ' '):
    X_src, X_tgt = [], []
    src_idx_word = {v:k for k ,v in src_word_idx.iteritems()}
    src_idx_word[0] = 'UNK'
    tgt_idx_word = {v:k for k ,v in tgt_word_idx.iteritems()}
    tgt_idx_word[0] = 'UNK'
    for i,f in enumerate(listdir(path)):
        bi_sents = open(join(path, f)).read().decode('utf-8').split(' ||| ')
        X_src.append(toks2idx(bi_sents[1].split(), src_word_idx, src_maxlen))
        X_tgt.append(toks2idx(bi_sents[0].split(), tgt_word_idx, tgt_maxlen))

        if i % 100 == 0:
            print 'Original Pair: ', open(join(path, f)).read()
            src_toks = [src_idx_word[i] for i in toks2idx(bi_sents[1].split(), src_word_idx, src_maxlen)]
            tgt_toks = [tgt_idx_word[i] for i in toks2idx(bi_sents[0].split(), tgt_word_idx, tgt_maxlen)]
            print 'Processed Pair: ', ' '.join(src_toks).encode('utf-8'), ' ||| ', ' '.join(tgt_toks).encode('utf-8')

    return pad_sequences(X_src, src_maxlen, dtype='int32'), pad_sequences(X_tgt, tgt_maxlen, dtype='int32')

def sample_data(X, y, size):
    idx = np.random.permutation(np.arange(X.shape[0]))
    return X[idx[:size],:], y[idx[:size]]

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip()

def load_text_vec(fname, vocab, splitter=' ', ext_num=20000):
    """
    Loads dx1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    ext_count = 0
    with open(fname, "r") as f:      
        # header = f.readline()
#        vocab_size, layer1_size = map(int, header.split())
        vocab_size = file_len(fname)
        layer1_size = None
        
        for line in f:
            ss = line.split(' ')
            if len(ss) <= 3:
                continue
            word = ss[0].decode('utf-8', 'ignore')
            dims = ' '.join(ss[1:]).strip().split(splitter)
            if layer1_size is None:
                layer1_size = len(dims)
                # print dims
                print "reading word2vec at vocab_size:%d, dimension:%d" % (vocab_size, layer1_size)
            if word in vocab:
                word_vecs[word] = np.fromstring(' '.join(dims), dtype='float32', count=layer1_size, sep=' ')

            elif ext_count < ext_num: # add this word to vocabulary
                ext_count += 1
                word_vecs[word] = np.fromstring(' '.join(dims), dtype='float32', count=layer1_size, sep=' ')

    return vocab_size, word_vecs, layer1_size

def doc2idx(docs, word_idx_map, maxlen = 400, num_clas = 2):
    y = np.zeros((len(docs), num_clas))
    X = []
    truncate_num = 0
    for t, datum in enumerate(docs):
        for k in datum['catgy']:
            y[t,k] = 1
        x = []
        for i,w in enumerate(datum['text'].split()):
            if i >= maxlen:
                truncate_num += 1
                break
            if w in word_idx_map:
                x.append(word_idx_map[w])
            else:
                x.append(0)
        X.append(x)
    print "truncate %d docs" % truncate_num
    return pad_sequences(X, maxlen, dtype='int32'), y
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            # word_vecs[word] = np.random.uniform(-0.25,0.25,k) 
            word_vecs[word] = np.zeros(k) 

def truncate_vocab(vocab, nb_words):
    new_vocab = defaultdict(float)
    i = 0
    for w in sorted(vocab, key=vocab.get, reverse=True):
        new_vocab[w] = vocab[w]
        i+=1
        if i >= nb_words:
            break
    return new_vocab

def build_data(data_folder, clean_string=False):
    """
    build data for text classification dataset
    assuming each folder in data_folder contains document(file) under the category
    """
    vocab = defaultdict(float)
    catgy_ind = defaultdict(int)
    test_docs = []
    
    # find all categories
    docDict = dict()
    catgy_idx = 0
    for topic in listdir(data_folder):
        if isdir(join(data_folder, topic)):
            print topic
            catgy_ind[topic] = catgy_idx
            catgy_idx += 1

            for docName in listdir(join(data_folder, topic)):
                # print 'processing %s...' % (join(data_folder, topic, docName))
                doc = open(join(data_folder, topic, docName)).read().decode('utf-8')
                docId = docName
                if clean_string:
                    orig_doc = clean_str(doc)
                else:
                    orig_doc = doc
                
                words = set(orig_doc.split())
                
                if not docId in docDict:
                    for word in words:
                        vocab[word] += 1
                    datum  = {"text": orig_doc,                             
                              "num_words": len(orig_doc.split()),
                              "catgy": [catgy_ind[topic]],
                              "Id": docId
                              }
                    docDict[docId] = datum  
                else:
                    docDict[docId]['catgy'].append(catgy_ind[topic])
            
    for datum in docDict.values():
        test_docs.append(datum)

    return test_docs, vocab, catgy_ind

def read_catgy_text(data_folder, w2v_file, maxlen=400, nb_words=40000):
    print "loading data...",        
    print "data_folder:", data_folder
    all_docs, vocab, catgy_ind = build_data(data_folder, clean_string=False)
    print "Data count:", len(all_docs)
    max_l = np.max(pd.DataFrame(all_docs)["num_words"])
    mean_l = np.mean(pd.DataFrame(all_docs)["num_words"])
    num_clas = np.max(catgy_ind.values()) + 1
    vocab = truncate_vocab(vocab, nb_words)
    print "data loaded!"

    print "number of documents: " + str(len(all_docs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "mean sentence length: " + str(mean_l)
    print "number of classes: " + str(num_clas)
    
#    print train_docs
    _, w2v, layer1_size = load_text_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    
    add_unknown_words(w2v, vocab, k=layer1_size)
    W, word_idx_map = get_W(w2v, k=layer1_size)
    # print word_idx_map
    # random.shuffle(all_docs)

    X_train, y_train = doc2idx(all_docs, word_idx_map, maxlen, num_clas)
    idx_word_map = {v:k for k ,v in word_idx_map.iteritems()}
    idx_word_map[0] = 'UNK'
    print "Original doc:", all_docs[0]["text"]
    print "Processed doc:", ' '.join([idx_word_map[i] for i in list(X_train[0,:])]).encode('utf-8')

    print "shape of processed data"
    print "X_train.shape:", X_train.shape
    print "y_train.shape:", y_train.shape

    print "dataset created!"
    return X_train, y_train, W, len(vocab)+1, maxlen, catgy_ind, word_idx_map