import numpy as np
import h5py
import sys
import cv2
import time
import random
import math
import pickle

from sketch_a_net import load_layers, load_pretrained, layers_meta

file_count = 0

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


def save_imgs(imgs, name, num_cols=10, pad=2):
    if len(imgs) == 0 or imgs[0] is None:
        return

    # h = imgs[0].shape[0]
    # w = imgs[0].shape[1]
    # h = 81
    # w = 81
    h = 105
    w = 105

    # Make cv img and write to it...
    num_rows = int(math.ceil(len(imgs) / float(num_cols)))
    total_img = np.ones(((h + (2 * pad)) * num_rows, (w + (2 * pad)) * num_cols))

    print('Start compile image')
    for i, img in enumerate(imgs):
        row = int(math.floor(i / float(num_cols)))
        col = i % num_cols
        if img is not None:
            img_pad = img.squeeze()
            h, w = img_pad.shape
            y = ((h + (pad * 2)) * row) + pad
            y_end = y + h
            x = ((w + (pad * 2)) * col) + pad
            x_end = x + w
            total_img[y:y_end, x:x_end] = img_pad
    print('Finished compile image')

    print('Start saving image')
    cv2.imwrite(name + '.png', total_img * 255)
    print('Finished saving image')


def get_pieces_for_layer(img, acts, layer_meta, pct=1, thresh=None, thresh_pct=1, top_threshold=None, top_thresholdPct=1):
    '''
    Given an image, its activations and a layer, return pieces of that image, their corresponding L2 normalized activation vectors, and their
    locations within the image
    '''
    global file_count
    layer_name, stride, f_size, padding = layer_meta

    if padding > 0:
        img_f = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    k, h, w, c = acts.shape
    acts_pieces = []
    img_pieces = []
    locations = []
    t_count = 0
    b_count = 0
    b_imgs = []
    t_imgs = []
    kept_imgs = []
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
            act_sum = np.sum(img_piece) / (f_size**2)
            if thresh is not None:
                if act_sum < thresh and random.random() > thresh_pct:
                    b_count += 1
                    b_imgs.append(img_piece)
                    continue
            if top_threshold is not None:
                if act_sum > top_threshold and random.random() > top_thresholdPct:
                    # print('act_sum', np.sum(img_piece), act_sum)
                    t_count += 1
                    t_imgs.append(img_piece)
                    continue
            kept_imgs.append(img_piece)

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

    # save_imgs(b_imgs, 'b_imgs' + str(file_count))
    # save_imgs(t_imgs, 't_imgs' + str(file_count))
    # save_imgs(kept_imgs, 'kept_imgs' + str(file_count))
    file_count += 1

    print('Filtered ' + str(t_count) + ' below and ' + str(b_count) + ' above of ' + str(len(acts_pieces) + t_count + b_count) + ' with threshold')

    return acts_pieces, img_pieces, locations


def get_pieces_for_layer2(img, acts, before, layer_meta, pct=1, thresh=None, thresh_to_keep=5):
    '''
    Given an image, its activations and a layer, return pieces of that image, their corresponding L2 normalized activation vectors, and their
    locations within the image (scale is for when you are checking grids of acts)
    '''
    layer_name, stride, f_size, padding = layer_meta

    if padding > 0:
        img_f = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    k, h, w, c = acts.shape
    # the selection to be processed at this layer might be larger than one act pixel and we need to adjust to that scale
    scale = before.shape[0]

    # how many strides to add to single act
    extra_strides = max(0, int((scale - f_size) / stride))

    acts_pieces = []
    img_pieces = []
    locations = []
    below_thresh_count = 0
    toss_count = 0
    keep_count = 0
    random.seed(30)

    below_thresh = []
    above_thresh = []

    to_keep = []
    for y in range(0, (h - extra_strides)):
        for x in range(0, (w - extra_strides)):
            # skip some percentage
            if random.random() > pct:
                continue

            # get section of original image
            x_start = x * stride
            x_end = x_start + f_size + (extra_strides * stride)
            y_start = y * stride
            y_end = y_start + f_size + (extra_strides * stride)
            img_piece = img[y_start: y_end, x_start: x_end]
            location = {'x': x_start, 'y': y_start}

            item = (x, y, img_piece, location)
            # skip if threshold for empty images not met
            if thresh is not None:
                if np.sum(img_piece) < thresh:
                        below_thresh.append(item)
                        continue
            to_keep.append(item)

    if len(below_thresh) > thresh_to_keep:
        to_keep_below_thresh = random.sample(below_thresh, thresh_to_keep)
        to_keep.extend(to_keep_below_thresh)
    else:
        to_keep.extend(below_thresh)

    for item in to_keep:
        x, y, img_piece, location = item
        # save image piece and location
        img_pieces.append(img_piece)
        locations.append(location)

        # save acts
        acts_y_end = y + extra_strides + 1
        acts_x_end = x + extra_strides + 1
        # get area of activations
        acts_piece = acts[0, y:acts_y_end, x:acts_x_end, :]
        # get averages for each channel
        acts_piece = np.einsum('ijk->k', acts_piece)
        # L2 normalization
        feat_norm = np.sqrt(np.sum(acts_piece ** 2))
        if feat_norm > 0:
            acts_piece = acts_piece / feat_norm
        acts_pieces.append(acts_piece)

    print('Kept ' + str(len(to_keep)) + ' and filtered ' + str(len(below_thresh) - thresh_to_keep) + ' of ' + str(len(below_thresh)) + ' below threshold')

    return acts_pieces, img_pieces, locations


def get_pieces_for_img(layers, imgs, befores, layer_names, pcts, threshes, thresh_to_keeps):
    print('Dividing into pieces')
    start_time = time.time()

    acts_by_layer = []
    for i, img in enumerate(imgs):
        acts = get_layers_output(layers, [layer_names[i]], img)[0]
        acts_by_layer.append(acts)

    # for each layer, get the sub images and their corresponding activations
    acts_pieces_by_layer = []
    img_pieces_by_layer = []
    locations_by_layer = []
    for i, acts in enumerate(acts_by_layer):
        start_time_layer = time.time()
        acts_pieces, img_pieces, locations = get_pieces_for_layer2(imgs[i], acts, befores[i], layers_meta[i], pcts[i], threshes[i], thresh_to_keeps[i])

        acts_pieces_by_layer.append(acts_pieces)
        img_pieces_by_layer.append(img_pieces)
        locations_by_layer.append(locations)
        print('Divided L' + str(i + 1) + ' in --- %s seconds ---' % (time.time() - start_time_layer))

    print('Divided in --- %s seconds ---' % (time.time() - start_time))
    return acts_pieces_by_layer, img_pieces_by_layer, locations_by_layer


# def get_pieces_for_img2(layers, img, before, layer_name, layer_index, pct, thresh, thresh_to_keep):
#     acts = get_layers_output(layers, [layer_name], img)[0]
#     start_time_layer = time.time()
#     acts_pieces, img_pieces, locations = get_pieces_for_layer2(img, acts, before, layers_meta[layer_index], pct, thresh, thresh_to_keep)
#     print('Divided L' + str(layer_index + 1) + ' in --- %s seconds ---' % (time.time() - start_time_layer))

#     return acts_pieces, img_pieces, locations


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


def load_corpus(sample_rate, pcts, thresholds, thresholdPcts, top_thresholds, top_thresholdPcts):
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
            if pcts[i] > 0:
                print('loading layer ' + str(i))
                acts_pieces, img_pieces, locations = get_pieces_for_layer(img, acts, layers_meta[i], pcts[i], thresholds[i], thresholdPcts[i], top_thresholds[i], top_thresholdPcts[i])

                acts_pieces_by_layer[i].extend(acts_pieces)
                img_pieces_by_layer[i].extend(img_pieces)

    print('Loaded corpus in --- %s seconds ---' % (time.time() - start_time))
    for i in range(len(layer_names)):
        print('For L' + str(i + 1) + ' loaded ' + str(len(acts_pieces_by_layer[i])) + ' acts and ' + str(len(img_pieces_by_layer[i])) + ' imgs')
    return acts_pieces_by_layer, img_pieces_by_layer, layers, layer_names


def load_corpus2(sample_rate, thresh=None, thresh_pct=1, top_threshold=None, top_thresholdPct=1):
    print('Load corpus2')
    layers = load_layers('./data/model_without_order_info_224.mat')
    start_time = time.time()

    with open('./data/sketchrnn_corpus.txt', 'rb') as fp:
        imgs = pickle.load(fp)

        imgs_f = []

        t_count = 0
        b_count = 0
        b_imgs = []
        t_imgs = []
        kept_imgs = []

        for i, img in enumerate(imgs):
            if i % sample_rate == 0:
                h, w = img.shape
                stt = int(h * 0.25)
                end = int(h * 0.75)
                cutout_size = int(h * 0.5)
                img_seg = img[stt:end, stt:end]
                img_sum = np.sum(img_seg) / (cutout_size**2)
                if thresh is not None:
                    if img_sum < thresh and random.random() > thresh_pct:
                        b_count += 1
                        b_imgs.append(img_seg)
                        continue
                if top_threshold is not None:
                    if img_sum > top_threshold and random.random() > top_thresholdPct:
                        # print('act_sum', np.sum(img_piece), act_sum)
                        t_count += 1
                        t_imgs.append(img_seg)
                        continue

                kept_imgs.append(img_seg)
                # img = cv2.resize(img, (57, 57), interpolation=cv2.INTER_AREA)
                # img = img.reshape(57, 57, 1)
                img = img.reshape(105, 105, 1)
                imgs_f.append(img)


        save_imgs(b_imgs, 'b_imgs')
        save_imgs(t_imgs, 't_imgs')
        save_imgs(kept_imgs, 'kept_imgs')

        acts = []
        for img in imgs_f:
            act = get_layers_output(layers, ['conv4'], img)[0]
            act = act[0, 2, 2, :]
            acts.append(act)

        print('Loaded corpus in --- %s seconds ---' % (time.time() - start_time))
        print('Loaded ' + str(len(imgs_f)) + ' imgs and ' + str(len(acts)) + ' acts')
        return imgs_f, acts

