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

    def get_similar_before(self, img, befores, n):
        start_time = time.time()
        img_f = format(img)
        print('image to search', img_f.shape)
        # get acts at each layer
        # get imgs and acts for this img
        print('Get pieces of whole img')
        acts_pieces_by_layer, img_pieces_by_layer, locations_by_layer = get_pieces_for_img(self.layers, self.layer_names, img_f)
        print('Finished getting pieces of whole img')

        match_thresholds = []
        matches_by_layer = []
        match_locations_by_layer = []
        time_match = 0
        time_acts = 0
        for i, before in enumerate(befores[:2]):
            print('Getting acts for L' + str(i + 1))
            start_time_acts = time.time()
            before_f = format(before)
            before_act = get_layers_output(self.layers, [self.layer_names[i]], before_f)[0]
            time_acts += time.time() - start_time_acts
            print('Found acts in --- %s seconds ---' % (time.time() - start_time_acts))

            start_time_f = time.time()
            acts_pieces = acts_pieces_by_layer[i]
            img_pieces = img_pieces_by_layer[i]
            locations = locations_by_layer[i]
            print('Formatting in --- %s seconds ---' % (time.time() - start_time_f))

            print('Finding matches for L' + str(i + 1))
            start_time_match = time.time()

            # L2 normalization
            target = before_act[0, 0, 0, :]
            feat_norm = np.sqrt(np.sum(target ** 2))
            if feat_norm > 0:
                target = target / feat_norm

            # find n closest matches...
            error = []
            print('shapes', acts_pieces[0].shape, target.shape)
            for acts_piece in acts_pieces:
                error.append(np.sum((acts_piece - target) ** 2))
            sort_idx = np.argsort(error)
            n_safe = min(n, len(sort_idx))
            top_matches = []
            top_match_locations = []
            print('avg before error', np.mean(error))
            for j in range(n_safe):
                match_idx = sort_idx[j]
                print('before matches error', error[match_idx])
                top_matches.append(img_pieces[match_idx])
                top_match_locations.append(locations[match_idx])
            # if i == 1:
            #     save_bar(acts_pieces[sort_idx[0]], 'match_bar' + str(i))
            #     save_bar(target, 'target_bar' + str(i))
            #     save_bar(error, 'error_bar' + str(i))

            time_match += time.time() - start_time_match
            print('Matched in --- %s seconds ---' % (time.time() - start_time_match))
            print('Finished finding matches for L' + str(i + 1))

            # get the imgs and their locations
            matches_by_layer.append(top_matches)
            match_locations_by_layer.append(top_match_locations)

        print('Found all acts in --- %s seconds ---' % time_acts)
        print('Matched all layers in --- %s seconds ---' % time_match)
        print('Loaded matches in --- %s seconds ---' % (time.time() - start_time))
        return matches_by_layer, match_locations_by_layer


    def get_similar_after(self, afters, n):
        '''
        Given an image, get a number of similar images
        '''
        start_time_afters = time.time()
        top_matches_by_layer = []
        for i, after in enumerate(afters[:2]):
            print('Getting acts for afters for L' + str(i + 1))
            start_time_acts = time.time()
            after_f = format(after)
            after_act = get_layers_output(self.layers, [self.layer_names[i]], after_f)[0]
            print('Found acts in --- %s seconds ---' % (time.time() - start_time_acts))

            after_act = after_act[0, 0, 0, :]

            # L2 normalization
            feat_norm = np.sqrt(np.sum(after_act ** 2))
            if feat_norm > 0:
                after_act = after_act / feat_norm
            target = after_act

            start_time_lookup = time.time()
            indices, distances = self.ANN.get_nn(i, target, n)
            print('Found ANN matches in --- %s seconds ---' % (time.time() - start_time_lookup))
            top_matches = []
            for j, idx in enumerate(indices):
                print(idx)
                print('error', distances[j])
                top_match = self.imgs[i][idx]
                top_matches.append(top_match)
            top_matches_by_layer.append(top_matches)

            print('Found afters matches in --- %s seconds ---' % (time.time() - start_time_afters))
        return top_matches_by_layer


