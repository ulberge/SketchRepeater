from annoy import AnnoyIndex

# Size of the vectors at each layer of Sketch-A-Net
fs = [64, 128, 256, 256]

# File names used by preprocess script and this file
# search_type = 'manhattan'
# file_extension = 'manhattan_L'
search_type = 'manhattan'
file_extension = 'test_manhattan_L'


class ANN:
    '''Class wrapping and providing API to Spotify ANNOY data structure'''
    def __init__(self):
        # Build the ANNs
        self.u4 = AnnoyIndex(fs[3], search_type)
        self.u4.load('./data/' + file_extension + '4.ann')
        self.u2 = AnnoyIndex(fs[1], search_type)
        self.u2.load('./data/' + file_extension + '2.ann')

    def get_nn(self, v_to_match, layer_index, n):
        '''Find approximate matches for the provided activation vector

        Keyword arguments:
            v_to_match -- vector to match
            layer_index -- index of layer to match at, could be 2 or 4
            n -- number of matches to return

        Returns:
            Returns a list of corpus indexes for matches
        '''
        if layer_index == 4:
            neighbors = self.u4.get_nns_by_vector(v_to_match, n, search_k=500, include_distances=False)
            return neighbors
        else:
            neighbors = self.u2.get_nns_by_vector(v_to_match, n, search_k=500, include_distances=False)
            return neighbors
