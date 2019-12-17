from annoy import AnnoyIndex
import random
import numpy as np
import pickle

from sketch_a_net import L2_concepts
from data_api import load_corpus, load_corpus2

fs = [64, 128, 256, 256, 256]
search_type = 'manhattan'
file_extension = 'manhattan_L'


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
        else:
            neighbors = self.u2.get_nns_by_vector(v, n, search_k=500, include_distances=False)
            return neighbors
