from PIL import Image
import base64
import re
import io
import numpy as np
import cv2


def get_img_array(data_url):
    '''Parse a dataURL into an image array
    '''
    meta, data = data_url.split(',')
    img_bytes = io.BytesIO(base64.b64decode(data))
    img = Image.open(img_bytes)
    img_arr = np.array(img)[:, :, 0]
    return img_arr


def get_data_url(img_arr):
    '''Parse an image array into a dataURL
    '''
    img_f = (1 - img_arr.squeeze()) * 255
    img_f = img_f.astype(np.uint8)
    pil_img = Image.fromarray(img_f)
    buff = io.BytesIO()
    pil_img.save(buff, format='PNG')
    img_data_url = base64.b64encode(buff.getvalue()).decode('utf-8')
    return img_data_url


def save_img(img, name):
    '''Save the given img as a file with the given name
    '''
    img = cv2.normalize(img, None, alpha=0, beta=255,
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)
    cv2.imwrite(name, img)


def format(img):
    img = img / 255.0
    img = 1 - img
    h, w = img.shape
    img = np.asarray(img).astype(np.float32).reshape((h, w, 1))
    return img


def normalize_L2(vec):
    # L2 normalization
    norm = np.sqrt(np.sum(vec ** 2))
    if norm > 0:
        vec = vec / norm
    return vec


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
