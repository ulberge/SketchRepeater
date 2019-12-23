import numpy as np
import cv2
import time
import random
from ann import ANN, file_extension
import pickle

from data_api import get_before_match_options
from sketch_a_net import load_layers, layers_meta, get_layers_output
from helpers import normalize_L2, overlapsInList, get_np_arr


class Repeater:
    '''Class for fetching suggested actions.'''
    def __init__(self):
        # load the layers of sketch-a-net with pretrained weights
        self.layers = load_layers('./data/model_without_order_info_224.mat')

        # load the corpuses of mark matches at layers 2 and 4
        # layer 2 matches low level marks like lines and
        # layer 4 matches higher levels marks like closed forms.
        with open('./data/corpus' + file_extension + '4.txt', 'rb') as fp:
            imgs, acts = pickle.load(fp)
            self.imgs4 = imgs
        with open('./data/corpus' + file_extension + '2.txt', 'rb') as fp:
            imgs, acts = pickle.load(fp)
            self.imgs2 = imgs

        # init the approximate nearest neighbors class for mark matching
        # the results from this can be fetched from self.imgs*
        self.ANN = ANN()

        # the layers for this repeater from sketch-a-net
        self.layer_names = ['conv1', 'conv2', 'conv3', 'conv4']

    def get_suggested_actions(self, befores, mark, afters, imgs):
        '''Fetches suggestions corresponding to each layer.

        Keyword arguments:
            befores -- image sections before mark at sizes appropriate to each layer
            mark -- the mark made
            afters -- image sections after mark at sizes appropriate to each layer
            imgs -- the current state of each AI canvas

        Returns:
            Returns a list of actions with the mark to be made and the location
            to make it.
        '''

        # find matches for the before images within the imgs and for the
        # marks from the corpus of marks.
        before_matches_by_layer = self.__get_before_matches(imgs, befores, 3)

        # make combinations of the marks with the before matches. Get mark
        # matches at layer 2 and 4 to represent low and high concepts
        mark_matches2, mark_matches4 = self.__get_mark_matches(mark, 3)

        # compare this with the afters and choose the closest matches to
        # return as suggested actions to be taken by the AIs.
        shape_to_match = mark.shape[:2]
        best_actions_by_layer = self.__get_best_actions(shape_to_match, before_matches_by_layer, mark_matches2, mark_matches4, afters)

        return best_actions_by_layer

    def __get_before_matches(self, imgs, befores, n):
        '''Get matches from images that are similar to the before images.

        Keyword arguments:
            imgs -- the current state of the AI images
            befores -- the before state where the mark was made on the human canvas at sizes appropriate for layer
            n -- number of matches to fetch

        Returns:
            Returns a list (by layer) of lists of before matches
        '''
        start_time = time.time()

        # the percent of randomly selected pieces of the image to search.
        # higher values will make it slower, there are a lot more strides
        # at lower levels.
        pcts = [0.0001, 0.001, 0.001, 0.001]

        # find the best matches for each layer
        # limit to first 3, because issue finding matches for L4
        before_matches_by_layer = []
        for i in range(0, 3):
            before = befores[i]
            shape_to_match = before.shape
            img_pieces, acts, locations = get_before_match_options(imgs[i], shape_to_match, pcts[i], self.layers, self.layer_names[i])

            # get the activations for this before image
            before_act = get_layers_output(self.layers, [self.layer_names[i]], before)[0]
            before_act = np.einsum('ijkc->c', before_act)
            target = normalize_L2(before_act)

            # find n closest matches from chopped up image...
            error = []
            for i, act in enumerate(acts):
                # use Euclidean distance
                act = normalize_L2(act)
                error.append(np.sum((act - target) ** 2))
            sort_idx = np.argsort(error)

            # get top matches that hopefully do not overlap
            n_safe = min(n, len(sort_idx))
            top_matches = []
            for j in range(n_safe):
                match_idx = sort_idx[j]
                location = locations[match_idx]
                img_piece = img_pieces[match_idx]
                match = [img_piece, location]
                # ignore matches that overlap matches we already found
                if not overlapsInList(match, top_matches):
                    top_matches.append(match)

            # get the imgs and their locations
            before_matches_by_layer.append(top_matches)

        # TODO: Fix issue with L4!
        # matching does not work well at L4 for some reason
        # for now, copy L3 matches to L4 and resize
        before_matches_L4 = []
        img_L3 = imgs[2]
        layer_name3, stride3, f_size3, padding3 = layers_meta[2]
        layer_name4, stride4, f_size4, padding4 = layers_meta[3]
        padding = (f_size4 - f_size3) / 2
        for before_match in before_matches_by_layer[2]:
            before_match_img, location = before_match
            x, y = location['x'] - padding, location['y'] - padding
            end_x, end_y = x + f_size4, y + f_size4
            location = {'x': x, 'y': y}
            before_match_img = img_L3[x:end_x, y:end_y]
            h, w, c = before_match_img.shape
            before_match_img = cv2.copyMakeBorder(before_match_img, 0, (f_size4 - h), 0, (f_size4 - w), cv2.BORDER_CONSTANT, value=[0., 0., 0.])
            before_matches_L4.append((before_match_img, location))
        before_matches_by_layer.append(before_matches_L4)

        print('Fetched before matches in --- %s seconds ---' % (time.time() - start_time))

        return before_matches_by_layer

    def __get_mark_matches(self, mark_to_match, n):
        '''Returns the top n matches to the given mark at layers 2 and 4 from the corpus. We could do every level, but this is faster and is meant to represent low (lines) and high level (closed forms) concepts.

        Keyword arguments:
            mark_to_match -- the mark made as an np array
            n -- number of matches to return

        Returns:
            Returns two lists of mark matches (from layer 2 and layer 4)
        '''
        start_time = time.time()

        f_size2 = layers_meta[1][2]
        mark2 = cv2.resize(mark_to_match, (f_size2, f_size2), interpolation=cv2.INTER_AREA)
        mark2_f = get_np_arr(mark2)
        mark2_acts = get_layers_output(self.layers, ['conv2'], mark2_f)[0]
        mark2_acts = mark2_acts[0, 0, 0, :]
        target2 = normalize_L2(mark2_acts)
        indices2 = self.ANN.get_nn(target2, 2, n)

        # get images corresponding to indices
        top_matches2 = []
        for idx in indices2:
            top_match = cv2.normalize(self.imgs2[idx], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            top_matches2.append(top_match)

        f_size4 = layers_meta[3][2]
        mark4 = cv2.resize(mark_to_match, (f_size4, f_size4), interpolation=cv2.INTER_AREA)
        mark4_f = get_np_arr(mark4)
        mark4_acts = get_layers_output(self.layers, ['conv4'], mark4_f)[0]
        mark4_acts = mark4_acts[0, 2, 2, :]
        target4 = normalize_L2(mark4_acts)
        indices4 = self.ANN.get_nn(target4, 4, n)

        # get images corresponding to indices
        top_matches4 = []
        for idx in indices4:
            top_match = cv2.normalize(self.imgs4[idx], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            top_matches4.append(top_match)

        print('Fetched mark matches in --- %s seconds ---' % (time.time() - start_time))

        return top_matches2, top_matches4

    def __get_before_mark_combinations(self, layer_index, shape_to_match, before_matches, mark_matches):
        '''Make combinations of the marks with the before matches.

        Keyword arguments:
            layerIndex -- the layer these combinations are for
            shape_to_match -- the size of the original shape to match
            before_matches -- the before images to use for the combinations
            mark_matches -- the marks to use for the combinations

        Returns:
            Returns a list of actions. Each action is a tuple of length 4 with the resulting combination, the mark made, the before piece image, and the location on the overall canvas.
        '''
        actions = []
        for before_match in before_matches:
            before_match_img, location = before_match
            # Try some random selection of the marks on before
            marks = random.sample(mark_matches, min(2, len(mark_matches)))
            for mark in marks:
                after = before_match_img.copy().squeeze()

                # resize mark to match original
                mark = cv2.resize(mark, shape_to_match,
                                  interpolation=cv2.INTER_AREA)

                # make line weights equal, its a problem on small images
                if layer_index < 2:
                    kernel = np.ones((2, 2), np.uint8)
                    mark = cv2.erode(mark, kernel, iterations=1)
                    ret, mark = cv2.threshold(mark, 0.4, 1, 0)

                # pad outside of mark
                h_pad = before_match_img.shape[1] - mark.shape[1]
                h_pad_left = int(h_pad / 2)
                h_pad_right = h_pad - h_pad_left
                v_pad = before_match_img.shape[0] - mark.shape[0]
                v_pad_top = int(v_pad / 2)
                v_pad_bottom = v_pad - v_pad_top
                if h_pad >= 0 or v_pad >= 0:
                    mark = cv2.copyMakeBorder(mark, v_pad_top, v_pad_bottom,
                                              h_pad_left, h_pad_right,
                                              cv2.BORDER_CONSTANT,
                                              value=[0., 0., 0.])
                else:
                    mark = cv2.resize(mark, before_match_img.shape[:2],
                                      interpolation=cv2.INTER_AREA)

                # add mark to before
                after += mark
                actions.append((after, mark, before_match_img, location))

        return actions

    def __get_best_actions(self, shape_to_match, before_matches_by_layer, mark_matches2, mark_matches4, afters):
        '''Make combinations of the marks with the before matches. Compare this with the afters and choose the closest matches to return as suggested actions.

        Keyword arguments:
            shape_to_match -- the size of the original shape to match
            before_matches_by_layer -- the matches from before the mark
            mark_matches2 -- low level mark matches (from layer 2)
            mark_matches4 -- high level mark matches (from layer 4)
            afters -- after images to compare combinations against for sort

        Returns:
            Returns a list with the best action for each layer.
        '''
        start_time_actions = time.time()

        best_actions = []
        for i, before_matches in enumerate(before_matches_by_layer):
            mark_matches = mark_matches2
            if i >= 2:
                mark_matches = mark_matches4
            actions = self.__get_before_mark_combinations(i, shape_to_match, before_matches, mark_matches)

            # get best option for this before by comparing to after
            actions = self.__sort_options(afters[i], actions, i)
            if len(actions) > 0:
                best_action = actions[0]
                best_actions.append(best_action)
            else:
                best_actions.append(None)

        print('Fetched best actions in --- %s seconds ---' %
              (time.time() - start_time_actions))

        return best_actions

    def __sort_options(self, after, options, layer_index):
        '''Sort the options (combinations of befores and marks) by comparing their activations with the after result

        Keyword arguments:
            after -- image for basis of comparison of the combinations
            options -- combinations of befores and marks to sort
            layer_index -- layer at which we are judging everything

        Returns:
            Returns a list of the options sorted by how closely their activations match the after image.
        '''
        start_time = time.time()

        # get activations for after at layer
        after_act = get_layers_output(self.layers, [self.layer_names[layer_index]], after)[0]
        after_act = np.einsum('ijkc->c', after_act)

        target = normalize_L2(after_act)

        # get activations for options at layer
        result_acts = []
        for option in options:
            result = option[0]
            result_f = get_np_arr(result)
            result_act = get_layers_output(self.layers, [self.layer_names[layer_index]], result_f)[0]
            result_act = np.einsum('ijkc->c', result_act)
            result_act = normalize_L2(result_act)
            result_acts.append(result_act)

        # calc difference between options and after for comparison
        error = []
        for i, result_act in enumerate(result_acts):
            error.append(np.sum((result_act - target) ** 2))

        # sort and return options in order of best match to after
        sort_idx = np.argsort(error)
        options_ordered = []
        for i in sort_idx:
            options_ordered.append(options[i])

        print('Found afters matches in --- %s seconds ---' % (time.time() - start_time))

        return options_ordered
