import numpy as np
import cv2
import time
import random
from ann import ANN, file_extension
import pickle

from data_api import load_corpus, get_pieces_for_img, get_layers_output
from sketch_a_net import load_layers, layers_meta
from helpers import get_data_url, get_img_arr, normalize_L2, overlapsInList


class Repeater:
    '''Class for fetching suggested actions
    '''
    def __init__(self):
        # load the layers of sketch-a-net with pretrained weights
        self.layers = load_layers('./data/model_without_order_info_224.mat')

        # load the corpus of mark matches
        with open('./data/corpus' + file_extension + '4.txt', 'rb') as fp:
            imgs, acts = pickle.load(fp)
            self.imgs4 = imgs
        with open('./data/corpus' + file_extension + '2.txt', 'rb') as fp:
            imgs, acts = pickle.load(fp)
            self.imgs2 = imgs

        # init the approximate nearest neighbors class for mark matching
        self.ANN = ANN()

        # the layers for this repeater from sketch-a-net
        self.layer_names = ['conv1', 'conv2', 'conv3', 'conv4']

    def get_suggested_actions(self, befores, mark, afters, imgs, n):
        '''Returns suggested actions correpsonding to each layer consisting of a mark and the location to make it.

        Keyword arguments:
        befores -- image sections before mark at sizes appropriate to each layer
        mark -- the mark made
        afters -- image sections after mark at sizes appropriate to each layer
        imgs -- the current state of each AI canvas
        n -- the number of suggested actions to fetch
        '''

        # find matches for the before images within the imgs and for the
        # marks from the corpus of marks.
        before_matches_by_layer = self.__get_before_matches(imgs, befores)

        # make combinations of the marks with the before matches. Get mark
        # matches at layer 2 and 4 to represent low and high concepts
        mark_matches2, mark_matches4 = self.__get_mark_matches(mark, 3)

        # compare this with the afters and choose the closest matches to
        # return as suggested actions to be taken by the AIs.
        best_actions_by_layer = self.__get_best_actions(before_matches_by_layer, mark_matches2, mark_matches4, afters)

        # format results
        result = {}
        for i in range(len(befores_matches)):
            best_action = best_actions_by_layer[i]
            befores = [before_match[0] for before_match in befores_matches[i]]
            result[i] = {
                'location': best_action[3],
                'mark': get_data_url(best_action[1])
            }

        return result

    def __get_before_matches(self, imgs, befores):
        '''Get matches from images that are similar to the before images.

        Keyword arguments:
        imgs -- the current state of the AI images
        befores -- the before state where the mark was made on the human canvas
        at sizes appropriate for layer
        '''
        start_time = time.time()

        # format the dataURLs into image arrays
        befores_f = map(get_img_array, befores)
        befores_f = map(format, befores_f)
        imgs_f = map(get_img_array, imgs)
        imgs_f = map(format, imgs_f)

        # get pieces of the image to search at different layers and their activations
        pcts = [0.05, 0.4, 0.5, 1]
        threshes = [0.1, 0.1, 0.1, 0.1]
        thresh_to_keeps = [20, 50, 50, 50]
        acts_pieces_by_layer, img_pieces_by_layer, locations_by_layer = get_pieces_for_img(self.layers, imgs_f, befores, self.layer_names, pcts, threshes, thresh_to_keeps)

        # find the best matches for each layer
        # limit to first 3, because issue finding matches for L4
        before_matches_by_layer = []
        for i, before in enumerate(befores_f[:3]):
            # get the activations for this before image
            before_act = get_layers_output(self.layers, [self.layer_names[i]], before)[0]
            before_act = np.einsum('ijkc->c', before_act)
            target = normalize_L2(before_act)

            # find n closest matches from chopped up image...
            acts_pieces = acts_pieces_by_layer[i]
            error = []
            for i, acts_piece in enumerate(acts_pieces):
                # use Euclidean distance
                acts_piece = normalize_L2(acts_piece)
                error.append(np.sum((acts_piece - target) ** 2))
            sort_idx = np.argsort(error)

            # get top matches that hopefully do not overlap
            img_pieces = img_pieces_by_layer[i]
            locations = locations_by_layer[i]
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
        img_L3 = format(imgs_f[2])
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

    def __get_mark_matches(self, mark, n):
        '''Returns the top n matches to the given mark at layers 2 and 4 from the corpus. We could do every level, but this is faster and is meant to represent low (lines) and high level (closed forms) concepts.

        Keyword arguments:
        mark -- the mark made
        '''
        start_time = time.time()

        # format the dataURLs into image arrays
        mark_to_match = get_img_array(mark)

        f_size2 = layers_meta[1][2]
        mark2 = cv2.resize(mark, (f_size2, f_size2), interpolation=cv2.INTER_AREA)
        mark2_f = format(mark2)
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
        mark4 = cv2.resize(mark, (f_size4, f_size4), interpolation=cv2.INTER_AREA)
        mark4_f = format(mark4)
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

    def __get_before_mark_combinations(self, layerIndex, before_matches, mark_matches):
        '''Make combinations of the marks with the before matches. Compare this with the afters and choose the closest matches to return as suggested actions.'''
        actions = []
        for before_match in before_matches:
            before_match_img, location = before_match
            # Try some random selection of the marks on before
            marks = random.sample(mark_matches, 2)
            for mark in marks:
                after = before_match_img.copy().squeeze()

                # resize mark to match largest original mark dim
                dim_to_match = max(mark_to_match.shape[0], mark_to_match.shape[1])
                mark = cv2.resize(mark, (dim_to_match, dim_to_match),
                                  interpolation=cv2.INTER_AREA)

                # make line weights equal, its a problem on small images
                if layerIndex < 2:
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

    def __get_best_actions(self, before_matches_by_layer, mark_matches2,
                         mark_matches4, afters):
        '''Make combinations of the marks with the before matches. Compare this with the afters and choose the closest matches to return as suggested actions.'''
        start_time_actions = time.time()

        # format the dataURLs into image arrays
        afters_f = map(get_img_array, afters)

        best_actions = []
        for i, before_matches in enumerate(before_matches_by_layer):
            mark_matches = i < 2 ? mark_matches2: mark_matches4
            actions = self.__get_before_mark_combinations(i, before_matches, mark_matches)

            # get best option for this before by comparing to after
            actions = self.__sort_options(afters_f[i], actions, i)
            best_action = actions[0]
            best_actions.append(best_action)

        print('Fetched best actions in --- %s seconds ---' %
              (time.time() - start_time_actions))

        return best_actions

    def __sort_options(self, after, options, layer_index):
        '''
        Given an image, get a number of similar images
        '''
        start_time = time.time()

        # get activations for after
        after_f = format(after)
        after_act = get_layers_output(self.layers, [self.layer_names[layer_index]], after_f)[0]
        after_act = np.einsum('ijkc->c', after_act)

        target = normalize_L2(after_act)

        # get activations for options
        result_acts = []
        for option in options:
            result = option[0]
            result_f = format(result)
            result_act = get_layers_output(self.layers, [self.layer_names[layer_index]], result_f)[0]
            result_act = np.einsum('ijkc->c', result_act)
            result_act = normalize_L2(result_act)
            result_acts.append(result_act)

        # calc error
        error = []
        for i, result_act in enumerate(result_acts):
            error.append(np.sum((result_act - target) ** 2))

        # sort and return options in order
        sort_idx = np.argsort(error)
        options_ordered = []
        for i in sort_idx:
            options_ordered.append(options[i])

        print('Found afters matches in --- %s seconds ---' % (time.time() - start_time))

        return options_ordered
