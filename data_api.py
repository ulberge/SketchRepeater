import numpy as np
import h5py
import sys
import cv2
import time
import random

from sketch_a_net import load_layers, load_pretrained, layers_meta


def get_layers_output(layers, layer_names, img):
    '''
    Get the activations for the layers in layer_names for this img and this
    set of layers
    '''
    # format for tf
    curr = np.array([img, ])
    # execute the layers
    acts = []
    count = 0
    for layer in layers:
        # run layer
        curr = layer(curr)

        if layer.name in layer_names:
            acts.append(curr)
            count += 1

        if count == len(layer_names):
            return acts

    return acts


def get_pieces_for_layer(img, acts, layer_meta, pct=1, thresh=None, thresh_pct=1):
    '''
    Given an image, its activations and a layer, return pieces of that image, their corresponding L2 normalized activation vectors, and their
    locations within the image
    '''
    layer_name, stride, f_size, padding = layer_meta

    if padding > 0:
        img_f = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    k, h, w, c = acts.shape
    acts_pieces = []
    img_pieces = []
    locations = []
    t_count = 0
    random.seed(30)
    for y in range(0, h):
        for x in range(0, w):
            # skip some percentage
            if random.random() > pct:
                continue

            # get section of original image
            x_start = x * stride
            x_end = x_start + f_size
            y_start = y * stride
            y_end = y_start + f_size
            img_piece = img[y_start: y_end, x_start: x_end]

            # skip if threshold for empty images not met
            if thresh is not None:
                if np.sum(img_piece) < thresh and random.random() > thresh_pct:
                    t_count += 1
                    continue

            # save image piece and location
            img_pieces.append(img_piece)
            locations.append({'x': x_start, 'y': y_start})

            # save acts
            acts_piece = acts[0, y, x, :]
            # L2 normalization
            feat_norm = np.sqrt(np.sum(acts_piece ** 2))
            if feat_norm > 0:
                acts_piece = acts_piece / feat_norm
            acts_pieces.append(acts_piece)

    print('Filtered ' + str(t_count) + ' of ' + str(len(acts_pieces) + t_count) + ' with threshold')

    return acts_pieces, img_pieces, locations


def get_pieces_for_img(layers, layer_names, img):
    print('Dividing into pieces')
    start_time = time.time()

    acts_by_layer = get_layers_output(layers, layer_names, img)

    # for each layer, get the sub images and their corresponding activations
    acts_pieces_by_layer = []
    img_pieces_by_layer = []
    locations_by_layer = []
    percents = [0.3, 0.5, 1, 1, 1]
    thresh_percents = [0.1, 1, 1, 1, 1]
    for i, acts in enumerate(acts_by_layer):
        start_time_layer = time.time()
        acts_pieces, img_pieces, locations = get_pieces_for_layer(img, acts, layers_meta[i], percents[i], 0.03, thresh_percents[i])

        acts_pieces_by_layer.append(acts_pieces)
        img_pieces_by_layer.append(img_pieces)
        locations_by_layer.append(locations)
        print('Divided L' + str(i + 1) + ' in --- %s seconds ---' % (time.time() - start_time_layer))

    print('Divided in --- %s seconds ---' % (time.time() - start_time))
    return acts_pieces_by_layer, img_pieces_by_layer, locations_by_layer


def load_corpus_imgs(sample_rate):
    print('Loading images from corpus')
    # load images from dataset
    data = h5py.File('./data/dataset_without_order_info_224.mat', 'r')
    all_imgs = data['imdb']['images']['data']

    imgs = [all_imgs[idx] for idx in range(0, len(all_imgs), sample_rate)]

    # format images for Sketch-A-Net
    imgs_f = []
    for img in imgs:
        # resize and format image
        img = img.swapaxes(0, 2)
        img = img[16:241, 16:241, :]
        img = img / 255
        img = 1 - img
        imgs_f.append(img)
    imgs = imgs_f

    print('Chose ' + str(len(imgs)) + ' images')
    return imgs


def load_corpus(sample_rate, pcts, thresholds, thresholdPcts):
    print('Load corpus')
    layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    # load layers of model
    layers = load_layers('./data/model_without_order_info_224.mat')
    start_time = time.time()

    imgs = load_corpus_imgs(sample_rate)

    acts_pieces_by_layer = [[], [], [], [], []]
    img_pieces_by_layer = [[], [], [], [], []]

    for img in imgs:
        acts_by_layer = get_layers_output(layers, layer_names, img)

        for i, acts in enumerate(acts_by_layer):
            acts_pieces, img_pieces, locations = get_pieces_for_layer(img, acts, layers_meta[i], pcts[i], thresholds[i], thresholdPcts[i])

            acts_pieces_by_layer[i].extend(acts_pieces)
            img_pieces_by_layer[i].extend(img_pieces)

    print('Loaded corpus in --- %s seconds ---' % (time.time() - start_time))
    for i in range(len(layer_names)):
        print('For L' + str(i + 1) + ' loaded ' + str(len(acts_pieces_by_layer[i])) + ' acts and ' + str(len(img_pieces_by_layer[i])) + ' imgs')
    return acts_pieces_by_layer, img_pieces_by_layer, layers, layer_names

