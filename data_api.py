import numpy as np
import h5py
import sys
import cv2
import time
import random
import math
import pickle

from sketch_a_net import load_layers, load_pretrained, layers_meta, get_layers_output
from helpers import save_imgs, split_img_by_receptive_field


def get_before_match_options(img, shape_to_match, pct_to_keep, layers, layer_name):
    '''Return a filtered set of the image split into segments matching the shape given

    Keyword arguments:
        img -- image to split
        shape_to_match -- shape to match when splitting
        pct_to_keep -- percent of the image pieces to keep
        layers -- layers to use for calculating activations
        layer_name -- name of layer to get activations for

    Returns:
        Returns a list of img pieces, their activations, and their locations
    '''
    start_time = time.time()

    h, w, c = img.shape
    img_pieces = []
    locations = []
    acts = []
    ySteps = int(h - shape_to_match[0]) + 1
    xSteps = int(w - shape_to_match[1]) + 1
    for y in range(0, ySteps):
        for x in range(0, xSteps):
            # skip some percentage
            if random.random() > pct_to_keep:
                continue

            # get section of before image
            x_start = x
            x_end = x_start + shape_to_match[0]
            y_start = y
            y_end = y_start + shape_to_match[1]
            img_piece = img[y_start: y_end, x_start: x_end]

            img_pieces.append(img_piece)

            # record location of this piece
            location = {'x': x_start, 'y': y_start}
            locations.append(location)

            # get activations for piece at specified layer
            act = get_layers_output(layers, [layer_name], img_piece)[0]
            # get averages for each channel
            act_avg = np.einsum('ijkc->c', act)
            acts.append(act_avg)

    print('Divided in --- %s seconds ---' % (time.time() - start_time))
    return img_pieces, acts, locations


def load_corpus_imgs_for_TU_Berlin(sample_rate):
    '''Load images from the corpus, selecting one every sample_rate.
    '''
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


def load_corpus_for_layer_2(sample_rate, pct_to_keep, low_threshold=None, low_threshold_pct=1, top_threshold=None, top_threshold_pct=1):
    '''Load corpus of image pieces to use for low level mark matches

    Keyword arguments:
        sample_rate -- percent of images from corpus to keep
        pct_to_keep -- percent of image segments to keep
        low_threshold -- the threshold to filter mostly empty image pieces
        low_threshold_pct -- percent of pieces below the threshold to keep
        top_threshold -- the threshold to filter really busy image pieces
        top_threshold_pct -- percent of pieces above the top threshold to keep

    Returns:
        Returns a list of img pieces and a corresponding list with their activations at layer 2.
    '''
    start_time = time.time()

    # load the Sketch-A-Net layers to use for calculating activations
    layers = load_layers('./data/model_without_order_info_224.mat')

    # get images from TU Berlin corpus
    imgs = load_corpus_imgs_for_TU_Berlin(sample_rate)

    # keep a record of the images filtered by the bottom and top filters and those kept
    b_imgs = []
    t_imgs = []
    kept_imgs = []
    layer_name, stride, f_size, padding = layers_meta[1]

    for img in imgs:
        # split the image up into pieces matching this layer2 size
        img_segs = split_img_by_receptive_field(img, stride, f_size, padding)
        for i, img_seg in enumerate(img_segs):
            # skip some percentage
            if random.random() > pct_to_keep:
                continue

            # calculate how much of the image is markings
            img_sum = np.sum(img_seg) / (f_size**2)

            # check low threshold
            if low_threshold is not None:
                if img_sum < low_threshold and random.random() > low_threshold_pct:
                    b_imgs.append(img_seg)
                    continue

            # check high threshold
            if top_threshold is not None:
                if img_sum > top_threshold and random.random() > top_threshold_pct:
                    t_imgs.append(img_seg)
                    continue

            # keep image
            kept_imgs.append(img_seg)

    # save the filtered images to help with tuning of filters
    save_imgs(b_imgs, 'b_2_imgs_debug')
    save_imgs(t_imgs, 't_2_imgs_debug')
    save_imgs(kept_imgs, 'kept_2_imgs_debug')

    # calculate the activations for the kept images at layer 4
    acts = []
    for img in kept_imgs:
        act = get_layers_output(layers, ['conv2'], img)[0]
        # get averages for each channel
        act_avg = np.einsum('ijkc->c', act)
        acts.append(act_avg)

    print('Loaded corpus in --- %s seconds ---' % (time.time() - start_time))
    print('Loaded ' + str(len(kept_imgs)) + ' imgs and ' + str(len(acts)) + ' acts')
    return kept_imgs, acts


def load_corpus_for_layer_4(sample_rate, low_threshold=None, low_threshold_pct=1, top_threshold=None, top_threshold_pct=1):
    '''Load corpus of image pieces to use for high level mark matches

    Keyword arguments:
        sample_rate -- percent of images from corpus to keep
        low_threshold -- the threshold to filter mostly empty image pieces
        low_threshold_pct -- percent of pieces below the threshold to keep
        top_threshold -- the threshold to filter really busy image pieces
        top_threshold_pct -- percent of pieces above the top threshold to keep

    Returns:
        Returns a list of img pieces and a corresponding list with their activations at layer 4.
    '''
    start_time = time.time()

    # load the Sketch-A-Net layers to use for calculating activations
    layers = load_layers('./data/model_without_order_info_224.mat')

    layer4_size = layers_meta[3][2]

    # load the preprocessed sketch corpus (contains images from Google "Quick, Draw" and TU Berlin)
    with open('./data/sketchrnn_corpus.txt', 'rb') as fp:
        imgs = pickle.load(fp)

        # keep a record of the images filtered by the bottom and top filters and those kept
        b_imgs = []
        t_imgs = []
        kept_imgs = []

        for i, img in enumerate(imgs):
            if i % sample_rate == 0:
                # cutout the center of the image to test for threshold
                # this gives us a better filter
                h, w = img.shape
                stt = int(h * 0.25)
                end = int(h * 0.75)
                cutout_size = int(h * 0.5)
                img_seg = img[stt:end, stt:end]

                # calculate how much of the image is markings
                img_sum = np.sum(img_seg) / (cutout_size**2)

                # check low threshold
                if low_threshold is not None:
                    if img_sum < low_threshold and random.random() > low_threshold_pct:
                        b_imgs.append(img_seg)
                        continue

                # check high threshold
                if top_threshold is not None:
                    if img_sum > top_threshold and random.random() > top_threshold_pct:
                        t_imgs.append(img_seg)
                        continue

                # keep image
                img = img.reshape(layer4_size, layer4_size, 1)
                kept_imgs.append(img)

        # save the filtered images to help with tuning of filters
        save_imgs(b_imgs, 'b_4_imgs_debug')
        save_imgs(t_imgs, 't_4_imgs_debug')
        save_imgs(kept_imgs, 'kept_4_imgs_debug')

        # calculate the activations for the kept images at layer 4
        acts = []
        for img in kept_imgs:
            act = get_layers_output(layers, ['conv4'], img)[0]
            # get averages for each channel
            act_avg = np.einsum('ijkc->c', act)
            acts.append(act_avg)

        print('Loaded corpus in --- %s seconds ---' % (time.time() - start_time))
        print('Loaded ' + str(len(kept_imgs)) + ' imgs and ' + str(len(acts)) + ' acts')
        return kept_imgs, acts

