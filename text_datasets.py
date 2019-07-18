import csv
import glob
import io
import os
import shutil
import sys
import tarfile
import tempfile

import numpy

import chainer

from nlp_utils import make_vocab
from nlp_utils import tokenize
from nlp_utils import transform_to_array

URL_IMDB = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'


def download_imdb():
    '''
    Downloads data in a temporary directory.
    '''
    path = chainer.dataset.cached_download(URL_IMDB)
    tf = tarfile.open(path, 'r')
    # To read many files fast, tarfile is untared
    path = tempfile.mkdtemp()
    tf.extractall(path)
    return path


def read_imdb(path, shrink=1, fine_grained=False, char_based=False):
    fg_label_dict = {'1': 0, '2': 0, '3': 1, '4': 1,
                     '7': 2, '8': 2, '9': 3, '10': 3}

    dataset = []
    target = os.path.join(path, 'aclImdb', 'train', 'unsup', '*')
    for i, f_path in enumerate(glob.glob(target)):

        if i % shrink != 0:
            continue
        with io.open(f_path, encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()

        tokens = tokenize(text, char_based)
        dataset.append((tokens, -1))
    unsup = dataset

    def read_and_label(split, posneg, label):
        dataset = []
        target = os.path.join(path, 'aclImdb', split, posneg, '*')
        for i, f_path in enumerate(glob.glob(target)):

            if i % shrink != 0:
                continue
            with io.open(f_path, encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()

            tokens = tokenize(text, char_based)

            if fine_grained:
                # extract from f_path. e.g. /pos/200_8.txt -> 8
                label = fg_label_dict[f_path.split('_')[-1][:-4]]
                dataset.append((tokens, label))
            else:
                dataset.append((tokens, label))
        return dataset

    print('Reading test sentences')
    pos_test = read_and_label('test', 'pos', 1)
    neg_test = read_and_label('test', 'neg', 0)

    print('Reading training sentences')
    pos_dataset = read_and_label('train', 'pos', 1)
    neg_dataset = read_and_label('train', 'neg', 0)

    test = pos_test + neg_test
    train = pos_dataset[:10621] + neg_dataset[:10625]
    valid = pos_dataset[10621:] + neg_dataset[10625:]

    return train, valid, test, unsup


def get_imdb(vocab=None, shrink=1, fine_grained=False, char_based=False, max_vocab_size=None):
    tmp_path = download_imdb()
    # tmp_path = 'data'

    # print('read imdb')
    train, valid, test, unsup = read_imdb(tmp_path, shrink=shrink, fine_grained=fine_grained, char_based=char_based)

    shutil.rmtree(tmp_path)

    if vocab is None:
        # print('construct vocabulary based on frequency')
        vocab = make_vocab(train, max_vocab_size)
        print('vocab size', len(vocab))
        vocab = make_vocab(train + unsup, max_vocab_size)
        print('vocab size', len(vocab))

    train = transform_to_array(train, vocab)
    valid = transform_to_array(valid, vocab)
    test = transform_to_array(test, vocab)
    unsup = transform_to_array(unsup, vocab)


    return train, valid, test, unsup, vocab

