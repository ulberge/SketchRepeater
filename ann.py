from annoy import AnnoyIndex
import random
import numpy as np
import pickle

from sketch_a_net import L2_concepts
from data_api import load_corpus

fs = [64, 128, 256, 256, 256]
# fs = [64, len(L2_concepts), 256, 256, 256]
# file_extension = '_0'
# search_type = 'manhattan'
# file_extension = '_angular'
file_extension = '_angular2'
search_type = 'angular'


def convert_v_L2(v):
    # concept_acts = np.zeros(len(L2_concepts))
    # for i, c in enumerate(L2_concepts):
    #     tot = 0
    #     for idx in c:
    #         tot += v[idx]
    #     concept_acts[i] = tot
    # return concept_acts
    return v


class ANN:
    def __init__(self):
        self.u = []
        for i in range(5):
            u_i = AnnoyIndex(fs[i], search_type)
            u_i.load('./data/L' + str(i + 1) + file_extension + '.ann')
            self.u.append(u_i)

    def get_nn(self, layer, v, n):
        if layer == 1:
            v = convert_v_L2(v)

        return self.u[layer].get_nns_by_vector(v, n, search_k=10000, include_distances=True)


if __name__ == '__main__':
    # sample_rate = 30
    sample_rate = 120
    # steps = [10, 1, 100, 100, 100]
    pcts = [0.01, 0.4, 0, 0, 0]
    thresholds = [0.03, 0.03, 0.03, 0.03, 0.03]
    thresholdPcts = [0.1, 0.5, 0.03, 0.03, 0.03]
    # load images chopped up into pieces with acts for each at the layer
    acts_pieces_by_layer, img_pieces_by_layer, layers, layer_names = load_corpus(sample_rate, pcts, thresholds, thresholdPcts)

    for i in range(5):
        acts_pieces = acts_pieces_by_layer[i]
        print('Building ANN for L' + str(i + 1))
        t = AnnoyIndex(fs[i], search_type)  # Length of item vector that will be indexed
        for idx, piece in enumerate(acts_pieces):
            if i == 1:
                piece = convert_v_L2(piece)
            t.add_item(idx, piece)

        t.build(2000)
        t.save('./data/L' + str(i + 1) + file_extension + '.ann')
        print('Finished building ANN for L' + str(i + 1))

    # pickle dump the corpus
    with open('./data/corpus' + file_extension + '.txt', 'wb') as fp:
        to_save = [acts_pieces_by_layer, img_pieces_by_layer, layer_names]
        pickle.dump(to_save, fp)