from annoy import AnnoyIndex
import random
import numpy as np
import pickle

from sketch_a_net import L2_concepts
from data_api import load_corpus, load_corpus2

fs = [64, 128, 256, 256, 256]
search_type = 'manhattan'
file_extension = 'manhattan_L'

seen4 = []
seen2 = []

class ANN:
    def __init__(self):
        self.u4 = AnnoyIndex(fs[3], search_type)
        self.u4.load('./data/' + file_extension + '4.ann')
        self.u2 = AnnoyIndex(fs[1], search_type)
        self.u2.load('./data/' + file_extension + '2.ann')

    def get_nn(self, v, layer, n):
        if layer == 4:
            neighbors = self.u4.get_nns_by_vector(v, n, search_k=500, include_distances=False)
            return neighbors
            # filter seen and cut down to requested number
            # neighbors = filter(lambda neighbor: neighbor not in seen4, neighbors)
            # print(len(neighbors))
            # neighbors = neighbors[:n]
            # print(len(neighbors))
            # seen4.extend(neighbors)
            # return neighbors
        else:
            neighbors = self.u2.get_nns_by_vector(v, n, search_k=500, include_distances=False)
            return neighbors
            # filter seen and cut down to requested number
            # neighbors = filter(lambda neighbor: neighbor not in seen2, neighbors)
            # neighbors = neighbors[:n]
            # seen2.extend(neighbors)
            # return neighbors


        # if len(neighbors) >= n:
        #     neighbors = neighbors[:n]
        #     seen.extend(neighbors)
        #     return neighbors
        # else:
        #     # backup just in case
        #     neighbors = self.u.get_nns_by_vector(v, n, search_k=10000, include_distances=True)
        #     return neighbors


def load2():
    layer_index = 1
    layer_num = '2'
    pcts = [
        [0.1, 0, 0, 0, 0],
        [0, 0.1, 0, 0, 0],
        [0, 0, 0.1, 0, 0],
        [0, 0, 0, 0.1, 0],
        [0, 0, 0, 0.1, 0]
    ]
    pcts = pcts[layer_index]

    # # thresholds = [0.03, 0.008, 0.01, 0.03, 0.03]
    # # thresholdPcts = [0.1, 0, 0, 0.03, 0.03]

    # # top_thresholds = [0.5, 0.04, 0.04, 0.5, 0.5]
    # # top_thresholdPcts = [0.1, 0, 0, 0.03, 0.03]

    sample_rate = 50
    thresholds = [0.03, 0.008, 0.008, 0.008, 0.03]
    thresholdPcts = [0.1, 0, 0, 0, 0.03]

    top_thresholds = [0.5, 0.03, 0.06, 0.04, 0.5]
    top_thresholdPcts = [0.1, 0, 0, 0, 0.03]
    # load images chopped up into pieces with acts for each at the layer
    acts_pieces_by_layer, img_pieces_by_layer, layers, layer_names = load_corpus(sample_rate, pcts, thresholds, thresholdPcts, top_thresholds, top_thresholdPcts)

    imgs = []
    acts = []
    # sample_rate = 2
    # thresholds = [0.03, 0.008, 0.008, 0, 0.03]
    # thresholdPcts = [0.1, 0, 0, 0, 0.03]

    # top_thresholds = [0.5, 0.06, 0.06, 0.04, 0.5]
    # top_thresholdPcts = [0.1, 0, 0, 0, 0.03]
    # imgs, acts = load_corpus2(sample_rate, thresholds[layer_index], thresholdPcts[layer_index], top_thresholds[layer_index], top_thresholdPcts[layer_index])

    imgs.extend(img_pieces_by_layer[layer_index])
    acts.extend(acts_pieces_by_layer[layer_index])

    print('Building ANN for L' + layer_num + ' with ' + str(len(imgs)) + ' imgs')
    t = AnnoyIndex(fs[layer_index], search_type)  # Length of item vector that will be indexed
    for idx, act in enumerate(acts):
        t.add_item(idx, act)

    t.build(20000)
    t.save('./data/' + file_extension + layer_num + '.ann')
    print('Finished building ANN for L' + layer_num)

    # pickle dump the corpus
    with open('./data/corpus' + file_extension + layer_num + '.txt', 'wb') as fp:
        to_save = [imgs, acts]
        pickle.dump(to_save, fp)


def load4():
    layer_index = 3
    layer_num = '4'
    pcts = [
        [0.1, 0, 0, 0, 0],
        [0, 0.1, 0, 0, 0],
        [0, 0, 0.1, 0, 0],
        [0, 0, 0, 0.1, 0],
        [0, 0, 0, 0, 0.1]
    ]
    pcts = pcts[layer_index]

    sample_rate = 100
    thresholds = [0.03, 0.008, 0.008, 0.008, 0.03]
    thresholdPcts = [0.1, 0, 0, 0, 0.03]

    top_thresholds = [0.5, 0.03, 0.06, 0.04, 0.5]
    top_thresholdPcts = [0.1, 0, 0, 0, 0.03]
    # load images chopped up into pieces with acts for each at the layer
    acts_pieces_by_layer, img_pieces_by_layer, layers, layer_names = load_corpus(sample_rate, pcts, thresholds, thresholdPcts, top_thresholds, top_thresholdPcts)

    imgs = []
    acts = []
    sample_rate = 1
    thresholds = [0.03, 0.008, 0.008, 0, 0.03]
    thresholdPcts = [0.1, 0, 0, 0, 0.03]

    top_thresholds = [0.5, 0.06, 0.06, 0.1, 0.5]
    top_thresholdPcts = [0.1, 0, 0, 0, 0.03]
    imgs, acts = load_corpus2(sample_rate, thresholds[layer_index], thresholdPcts[layer_index], top_thresholds[layer_index], top_thresholdPcts[layer_index])

    imgs.extend(img_pieces_by_layer[layer_index])
    acts.extend(acts_pieces_by_layer[layer_index])

    print('Building ANN for L' + layer_num + ' with ' + str(len(imgs)) + ' imgs')
    t = AnnoyIndex(fs[layer_index], search_type)  # Length of item vector that will be indexed
    for idx, act in enumerate(acts):
        t.add_item(idx, act)

    t.build(20000)
    t.save('./data/' + file_extension + layer_num + '.ann')
    print('Finished building ANN for L' + layer_num)

    # pickle dump the corpus
    with open('./data/corpus' + file_extension + layer_num + '.txt', 'wb') as fp:
        to_save = [imgs, acts]
        pickle.dump(to_save, fp)


if __name__ == '__main__':
    # load2()
    load4()
