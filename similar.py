import numpy as np
import cv2
import time
import random
from ann import ANN, file_extension
import pickle
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

# from sklearn.neighbors import LSHForest

from data_api import load_corpus, get_pieces_for_img, get_layers_output
from sketch_a_net import load_layers


def format(img):
    img = img / 255.0
    img = 1 - img
    h, w = img.shape
    img = np.asarray(img).astype(np.float32).reshape((h, w, 1))
    return img


def save_img(img, name):
    img = cv2.normalize(1 - img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)
    cv2.imwrite(name, img)


def align_images_safe(fixed_img, img_to_align):
    # avg_cluster_img = imgs[0].squeeze()
    # for i, img in enumerate(imgs[1:]):
    #     avg_cluster_img_8U = np.uint8(avg_cluster_img * 255).squeeze()
    #     img_8U = np.uint8(img * 255).squeeze()
    #     # try to align img with avg
    #     img_aligned = img_8U
    img_aligned = img_to_align
    try:
        img_aligned = align_images(fixed_img, img_to_align)
    except cv2.error as e:
        # cannot align
        print('Alignment failure!', e)
        pass
    return img_aligned


def align_images(im1_gray, im2_gray):
    '''
    Modified From: https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    '''
    # Find size of image1
    sz = im1_gray.shape

    # Define the motion model
    # warp_mode = cv2.MOTION_EUCLIDEAN
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-1

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 1)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2_gray, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2_gray, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    return im2_aligned


# def save_bar(a, name):
#     if not plt.get_fignums():
#         plt.figure()
#     plt.bar(np.arange(len(a)), a)
#     plt.savefig(name + '.png')
#     plt.clf()


def overlap(r0, r1):
    r0c0x, r0c0y, r0c1x, r0c1y = r0
    r1c0x, r1c0y, r1c1x, r1c1y = r1

    # is a corner of r0 inside r1?
    if r0c0x >= r1c0x and r0c0x <= r1c1x and r0c0y >= r1c0y and r0c0y <= r1c1y:
        return True
    if r0c1x >= r1c0x and r0c1x <= r1c1x and r0c1y >= r1c0y and r0c1y <= r1c1y:
        return True

    if r1c0x >= r0c0x and r1c0x <= r0c1x and r1c0y >= r0c0y and r1c0y <= r0c1y:
        return True
    if r1c1x >= r0c0x and r1c1x <= r0c1x and r1c1y >= r0c0y and r1c1y <= r0c1y:
        return True

    return False


def overlapsInList(match, other_matches):
    img_piece, location = match
    # if this piece overlaps other match, skip
    piece_bounds = [
        location['x'], location['y'],
        location['x'] + img_piece.shape[1], location['y'] + img_piece.shape[0]
    ]

    for other_match in other_matches:
        img_piece2, location2 = other_match
        # if this piece overlaps other match, skip
        piece_bounds2 = [
            location2['x'], location2['y'],
            location2['x'] + img_piece2.shape[1], location2['y'] + img_piece2.shape[0]
        ]
        if overlap(piece_bounds, piece_bounds2):
            return True

    return False


class Repeater:
    def __init__(self):
        self.layers = load_layers('./data/model_without_order_info_224.mat')
        with open('./data/corpus' + file_extension + '.txt', 'rb') as fp:
            acts_pieces_by_layer, img_pieces_by_layer, layer_names = pickle.load(fp)
            self.layer_names = layer_names
            self.imgs = img_pieces_by_layer

        self.ANN = ANN()

        # print('Building LSHForest with n=' + str(len(self.acts)))
        # self.lshf = LSHForest()
        # self.lshf.fit(self.acts)
        # print('Finished building LSHForest')

    # def get_similar_before(self, img, layer_index, before, bounds, n):
    #     start_time = time.time()
    #     img_f = format(img)
    #     print('Shape of image to search', img_f.shape)

    #     # Get pieces of the image to search at different layers and their activations
    #     print('Get pieces of whole img')
    #     layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    #     pcts = [0.02, 0.08, 0.2, 0.2, 0.2]
    #     threshes = [0.1, 0.1, 0.1, 0.1, 0.1]
    #     thresh_to_keeps = [5, 5, 5, 5, 5]

    #     acts_pieces, img_pieces, locations = get_pieces_for_img2(self.layers, img_f, before, layer_names[layer_index], layer_index, pcts[layer_index], threshes[layer_index], thresh_to_keeps[layer_index])
    #     print('Finished getting pieces of whole img for L' + str(layer_index + 1))
    #     print(len(i) for i in acts_pieces)

    #     # Find the best matches
    #     print('Getting acts in provided change area for L' + str(i + 1))
    #     start_time_acts = time.time()
    #     before_f = format(before)
    #     before_act = get_layers_output(self.layers, [layer_names[i]], before_f)[0]
    #     before_act = np.einsum('ijkc->c', before_act)
    #     # L2 normalization
    #     target = before_act
    #     feat_norm = np.sqrt(np.sum(target ** 2))
    #     if feat_norm > 0:
    #         target = target / feat_norm
    #     print('Found acts in --- %s seconds ---' % (time.time() - start_time_acts))
    #     print('Shape of change area', before_act.shape)

    #     start_time_f = time.time()
    #     acts_pieces = acts_pieces_by_layer[i]
    #     img_pieces = img_pieces_by_layer[i]
    #     locations = locations_by_layer[i]
    #     print('Formatting in --- %s seconds ---' % (time.time() - start_time_f))

    #     print('Calcu matches for L' + str(i + 1))
    #     start_time_error = time.time()
    #     # find n closest matches...
    #     error = []
    #     for i, acts_piece in enumerate(acts_pieces):
    #         error.append(np.sum((acts_piece - target) ** 2))
    #     print('Calculated error in --- %s seconds ---' % (time.time() - start_time_error))

    #     # get top matches that hopefully do not overlap
    #     start_time_match = time.time()
    #     sort_idx = np.argsort(error)
    #     n_safe = min(n, len(sort_idx))
    #     top_matches = []
    #     for j in range(n_safe):
    #         match_idx = sort_idx[j]
    #         location = locations[match_idx]
    #         img_piece = img_pieces[match_idx]
    #         match = [img_piece, location]

    #         if not overlapsInList(match, top_matches):
    #             top_matches.append(match)

    #     print('Found matches in --- %s seconds ---' % (time.time() - start_time_match))
    #     print([i[1] for i in top_matches])

    #     # get the imgs and their locations
    #     matches_by_layer.append(top_matches)

    #     print('Loaded matches in --- %s seconds ---' % (time.time() - start_time))
    #     return matches_by_layer

    def get_similar_befores(self, imgs, befores, n):
        start_time = time.time()
        print('Shape of images to search', len(imgs))
        imgs_f = map(format, imgs)
        print('Shape of images to search', [img.shape for img in imgs_f])

        # Get pieces of the image to search at different layers and their activations
        print('Get pieces of whole img')
        layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        # pcts = [0.02, 0.08, 0.2, 0.2, 0.2]
        # threshes = [0.1, 0.1, 0.1, 0.1, 0.1]
        # thresh_to_keeps = [5, 5, 5, 5, 5]
        pcts = [0.1, 0.3, 0.5, 0.5, 0.5]
        threshes = [0.1, 0.1, 0.1, 0.1, 0.1]
        thresh_to_keeps = [50, 50, 50, 50, 50]

        acts_pieces_by_layer, img_pieces_by_layer, locations_by_layer = get_pieces_for_img(self.layers, imgs_f, befores, layer_names, pcts, threshes, thresh_to_keeps)
        print('Finished getting pieces of whole img')
        print([len(i) for i in acts_pieces_by_layer])
        print([i[0].shape for i in acts_pieces_by_layer])

        # Find the best matches for each layer
        matches_by_layer = []
        # For each layer
        for i, before in enumerate(befores):
            print('Getting acts in provided change area for L' + str(i + 1))
            start_time_acts = time.time()
            before_f = format(before)
            before_act = get_layers_output(self.layers, [layer_names[i]], before_f)[0]
            before_act = np.einsum('ijkc->c', before_act)
            # L2 normalization
            target = before_act
            feat_norm = np.sqrt(np.sum(target ** 2))
            if feat_norm > 0:
                target = target / feat_norm
            print('Found acts in --- %s seconds ---' % (time.time() - start_time_acts))
            print('Shape of change area', before_act.shape)

            start_time_f = time.time()
            acts_pieces = acts_pieces_by_layer[i]
            img_pieces = img_pieces_by_layer[i]
            locations = locations_by_layer[i]
            print('Formatting in --- %s seconds ---' % (time.time() - start_time_f))

            print('Calcu matches for L' + str(i + 1))
            start_time_error = time.time()
            # find n closest matches...
            error = []
            for i, acts_piece in enumerate(acts_pieces):
                # punish edge pieces to remove
                if (len(img_pieces[i]) != before.shape[0] or len(img_pieces[i][0]) != before.shape[1]):
                    error.append(9999999999)
                else:
                    error.append(np.sum((acts_piece - target) ** 2))
            print('Calculated error in --- %s seconds ---' % (time.time() - start_time_error))

            # get top matches that hopefully do not overlap
            start_time_match = time.time()
            sort_idx = np.argsort(error)
            n_safe = min(n, len(sort_idx))
            top_matches = []
            for j in range(n_safe):
                match_idx = sort_idx[j]
                location = locations[match_idx]
                img_piece = img_pieces[match_idx]
                match = [img_piece, location]

                if not overlapsInList(match, top_matches):
                    top_matches.append(match)

            print('Found matches in --- %s seconds ---' % (time.time() - start_time_match))
            print([i[1] for i in top_matches])

            # get the imgs and their locations
            matches_by_layer.append(top_matches)

        print('Loaded matches in --- %s seconds ---' % (time.time() - start_time))
        return matches_by_layer


    def get_similar_marks(self, mark, n):
        '''
        Given an image, get a number of similar images
        '''
        start_time = time.time()

        # resize mark to fit conv2
        orig_size = mark.shape[0]
        mark = cv2.resize(mark, (45, 45), interpolation=cv2.INTER_AREA)

        start_time_acts = time.time()
        mark_f = format(mark)
        mark_acts = get_layers_output(self.layers, ['conv2'], mark_f)[0]
        print('Should only have one dimension', mark_acts.shape)
        mark_acts = np.einsum('ijkc->c', mark_acts)
        # L2 normalization
        feat_norm = np.sqrt(np.sum(mark_acts ** 2))
        if feat_norm > 0:
            mark_acts = mark_acts / feat_norm
        target = mark_acts
        print('Found acts in --- %s seconds ---' % (time.time() - start_time_acts))

        start_time_lookup = time.time()
        indices, distances = self.ANN.get_nn(target, n)
        print('Found ANN matches in --- %s seconds ---' % (time.time() - start_time_lookup))
        top_matches = []
        for j, idx in enumerate(indices):
            print(idx)
            print('error', distances[j])
            top_match = self.imgs[1][idx]
            top_matches.append(top_match)

        print('Found matches to mark in --- %s seconds ---' % (time.time() - start_time))

        # upscale matches back to match original
        # top_matches = [cv2.resize(match, (orig_size, orig_size), interpolation=cv2.INTER_AREA) for match in top_matches]

        # erode/dilate other filters to try and make it match original stroke width better...

        return top_matches


    def sort_options(self, after, options, layer_index):
        '''
        Given an image, get a number of similar images
        '''
        start_time_afters = time.time()

        # Get activations for after
        start_time_acts = time.time()
        after_f = format(after)
        after_act = get_layers_output(self.layers, [self.layer_names[layer_index]], after_f)[0]
        after_act = np.einsum('ijkc->c', after_act)
        # L2 normalization
        target = after_act
        feat_norm = np.sqrt(np.sum(target ** 2))
        if feat_norm > 0:
            target = target / feat_norm
        print('Found after acts in --- %s seconds ---' % (time.time() - start_time_acts))

        # Get activations for options
        start_time_acts_options = time.time()
        option_acts = []
        for option in options:
            option_f = format(option)
            option_act = get_layers_output(self.layers, [self.layer_names[layer_index]], option_f)[0]
            option_act = np.einsum('ijkc->c', option_act)
            # L2 normalization
            feat_norm = np.sqrt(np.sum(option_act ** 2))
            if feat_norm > 0:
                option_act = option_act / feat_norm
            option_acts.append(option_act)
        print('Found options acts in --- %s seconds ---' % (time.time() - start_time_acts_options))

        # Calc error
        start_time_error = time.time()
        # find n closest matches...
        error = []
        for i, option_act in enumerate(option_acts):
            error.append(np.sum((option_act - target) ** 2))
        print('Calculated error in --- %s seconds ---' % (time.time() - start_time_error))

        # Sort and return options in order
        sort_idx = np.argsort(error)
        options_ordered = []
        for i in sort_idx:
            options_ordered.append(options[i])

        print('Found afters matches in --- %s seconds ---' % (time.time() - start_time_afters))
        return options_ordered


