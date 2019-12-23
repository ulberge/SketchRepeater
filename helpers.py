from PIL import Image
import base64
import re
import io
import numpy as np
import cv2
import math


def split_img_by_receptive_field(img, stride, f_size, padding):
    '''Given an image, split it into segments for the CNN info

    Keyword arguments:
        img -- image to split
        stride -- shift between theoretical receptive fields
        f_size -- theoretical receptive field size
        padding -- total padding added by this layer

    Returns:
        Returns a list of img pieces
    '''
    if padding > 0:
        img = cv2.copyMakeBorder(img, padding, padding, padding, padding,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])

    h, w, c = img.shape
    img_pieces = []
    for y in range(0, int((h - f_size) / stride) + 1):
        for x in range(0, int((w - f_size) / stride) + 1):
            # get section of original image
            x_start = x * stride
            x_end = x_start + f_size
            y_start = y * stride
            y_end = y_start + f_size
            img_piece = img[y_start: y_end, x_start: x_end]

            img_pieces.append(img_piece)

    return img_pieces


def get_img_arr(data_url):
    '''Parse a dataURL into an image array
    '''
    meta, data = data_url.split(',')
    img_bytes = io.BytesIO(base64.b64decode(data))
    img = Image.open(img_bytes)
    img_arr = np.array(img)[:, :, 0]
    return img_arr


def get_data_url(img_arr):
    '''Parse an image array into a dataURL'''
    img_f = (1 - img_arr.squeeze()) * 255
    img_f = img_f.astype(np.uint8)
    pil_img = Image.fromarray(img_f)
    buff = io.BytesIO()
    pil_img.save(buff, format='PNG')
    img_data_url = base64.b64encode(buff.getvalue()).decode('utf-8')
    return img_data_url


def save_imgs(imgs, name, num_cols=10, pad=2):
    '''Compile the images as a grid and save as one image'''
    if len(imgs) == 0 or imgs[0] is None:
        return

    h, w, c = imgs[0].shape

    num_rows = int(math.ceil(len(imgs) / float(num_cols)))

    # create empty image to store the images in
    total_img = np.ones(((h + (2 * pad)) * num_rows, (w + (2 * pad)) * num_cols))

    # insert images into big empty image
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

    # save image
    cv2.imwrite(name + '.png', total_img * 255)


def save_img(img, name):
    '''Save the given img as a file with the given name'''
    img = cv2.normalize(img, None, alpha=0, beta=255,
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)
    cv2.imwrite(name, img)


def get_np_arr(img):
    '''Format a loaded image from its default format to a np array'''
    img = img / 255.0
    img = 1 - img
    h = img.shape[0]
    w = img.shape[1]
    img = np.asarray(img).astype(np.float32).reshape((h, w, 1))
    return img


def normalize_L2(vec):
    '''Return a copy of the vector normalized (L2)'''
    norm = np.sqrt(np.sum(vec ** 2))
    if norm > 0:
        vec = vec / norm
    return vec


def overlap(r0, r1):
    '''Return True if the rectangles overlap, else False'''
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
    '''Return True if any of the matches overlap the bounds of a match in the list of matches, else False.'''
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
