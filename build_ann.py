from annoy import AnnoyIndex
import random
import numpy as np
import pickle

from data_api import load_corpus_for_layer_2, load_corpus_for_layer_4

from ann import fs, search_type, file_extension


def create2():
    sample_rate = 10000
    # sample_rate = 50
    pct_to_keep = 0.1
    low_threshold = 0.008
    low_threshold_pct = 0.001
    top_threshold = 0.03
    top_threshold_pct = 0.001

    # load images chopped up into pieces with acts for each at the layer
    imgs, acts = load_corpus_for_layer_2(sample_rate, pct_to_keep, low_threshold, low_threshold_pct, top_threshold, top_threshold_pct)

    print('Building ANN for L2 with ' + str(len(imgs)) + ' imgs')

    t = AnnoyIndex(fs[1], search_type)
    for idx, act in enumerate(acts):
        t.add_item(idx, act)

    t.build(20000)
    t.save('./data/' + file_extension + '2.ann')
    print('Finished building ANN for L2')

    # pickle dump the corpus
    with open('./data/corpus' + file_extension + '2.txt', 'wb') as fp:
        to_save = [imgs, acts]
        pickle.dump(to_save, fp)


def create4():
    sample_rate = 10000
    # sample_rate = 100
    low_threshold = 0.008
    low_threshold_pct = 0.001
    top_threshold = 0.04
    top_threshold_pct = 0.001

    # load images chopped up into pieces with acts for each at the layer
    imgs, acts = load_corpus_for_layer_4(sample_rate, low_threshold, low_threshold_pct, top_threshold, top_threshold_pct)

    print('Building ANN for L4 with ' + str(len(imgs)) + ' imgs')

    t = AnnoyIndex(fs[3], search_type)
    for idx, act in enumerate(acts):
        t.add_item(idx, act)

    t.build(20000)
    t.save('./data/' + file_extension + '4.ann')
    print('Finished building ANN for L4')

    # pickle dump the corpus
    with open('./data/corpus' + file_extension + '4.txt', 'wb') as fp:
        to_save = [imgs, acts]
        pickle.dump(to_save, fp)


if __name__ == '__main__':
    create2()
    create4()
