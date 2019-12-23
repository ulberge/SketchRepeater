import numpy as np
import cv2
import math
import struct
from struct import unpack
import cairocffi as cairo
import ndjson
import random
import pickle


def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break


def vector_to_raster(vector_images, side=28, line_diameter=16, max_padding=4, bg_color=(0,0,0), fg_color=(1,1,1)):
    """
    padding and line_diameter are relative to the original 256x256 image.
    """

    original_side = 256.
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)

    raster_images = []
    for vector_image in vector_images:
        rand = int(random.random() * max_padding)
        padding = rand**2
        ctx = cairo.Context(surface)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        # ctx.set_line_width(line_diameter * math.sqrt(rand))
        # ctx.set_line_width(line_diameter * rand)
        # print(rand, padding)
        ctx.set_line_width(line_diameter * 8 / (1 + ((20 - rand) / 6)))
        # scale to match the new size
        # add padding at the edges for the line_diameter
        # and add additional padding to account for antialiasing
        total_padding = padding * 2. + line_diameter
        new_scale = float(side) / float(original_side + total_padding)
        ctx.scale(new_scale, new_scale)
        ctx.translate(total_padding / 2., total_padding / 2.)

        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()

        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1,1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image)

    return raster_images


def load_from(type, num=400, size=105, max_padding=20, line_radius=2):
    print('loading ' + type)
    # load from file-like objects
    with open('./full_simplified_' + type + '.ndjson') as f:
        data = ndjson.load(f)

        vecs = []
        for d in data[:num]:
            vecs.append(d['drawing'])

        imgs = vector_to_raster(vecs, size, line_radius, max_padding)

        imgs_f = []
        for img in imgs:
            img_f = img.reshape(size, size)
            img_f = cv2.normalize(img_f, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            imgs_f.append(img_f)

        return imgs_f


def save_imgs(imgs, name, num_cols=10, pad=2):
    if len(imgs) == 0 or imgs[0] is None:
        return

    h = imgs[0].shape[0]
    w = imgs[0].shape[1]

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
    cv2.imwrite(name + '.png', (1 - total_img) * 255)
    print('Finished saving image')


def rotate_random(img, amt):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    angle = (random.random() * amt * 2) - amt
    # angle = 15
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (h, w))
    return rotated


if __name__ == '__main__':
    all_imgs = []

    num = 40
    lines = load_from('line', num, max_padding=4)
    all_imgs.extend(lines)

    to_rotates = []
    to_rotates.extend(load_from('line', num, max_padding=4))
    to_rotates.extend(load_from('zigzag', 40))
    to_rotates.extend(load_from('triangle', 40))
    to_rotates.extend(load_from('squiggle', 20))
    rotated = []
    for img in to_rotates:
        img_r = rotate_random(img, 45)
        rotated.append(img_r)
    all_imgs.extend(rotated)

    all_imgs.extend(load_from('squiggle', 20))
    all_imgs.extend(load_from('zigzag', num))
    all_imgs.extend(load_from('circle', num))
    all_imgs.extend(load_from('triangle', num))
    all_imgs.extend(load_from('square', num))
    all_imgs.extend(load_from('hexagon', num))
    all_imgs.extend(load_from('octagon', num))
    all_imgs.extend(load_from('pillow', num))

    # squares, circles, cut in half -> curves, ends
    # zigzags cut in half both ways -> multi curves?
    # clouds cut in half both ways -> curves
    wholes = []
    num = 15
    wholes.extend(load_from('circle', num))
    wholes.extend(load_from('square', num))
    wholes.extend(load_from('zigzag', num))
    wholes.extend(load_from('cloud', num))

    halves = []
    for whole in wholes:
        # cut in halves
        left = whole[:, :53]
        right = whole[:, 52:]
        top = whole[:53, :]
        bottom = whole[52:, :]

        pad = 26
        left = cv2.copyMakeBorder(left, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0., 0., 0.])
        right = cv2.copyMakeBorder(right, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0., 0., 0.])

        top = cv2.copyMakeBorder(top, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0., 0., 0.])
        bottom = cv2.copyMakeBorder(bottom, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0., 0., 0.])

        halves.append(left)
        halves.append(right)
        halves.append(top)
        halves.append(bottom)

        # cv2.imwrite('test_left.png', left)
        # cv2.imwrite('test_right.png', right)
        # cv2.imwrite('test_top.png', top)
        # cv2.imwrite('test_bottom.png', bottom)
    all_imgs.extend(halves)

    # squares, circles, cut in quarters and rotate -> corners, curved corners
    wholes = []
    wholes.extend(load_from('circle', num, 210, line_radius=1))
    wholes.extend(load_from('square', num, 210, line_radius=1))
    corners = []
    for whole in wholes:
        # cut in halves
        c_size = 105
        r_size = 105

        top_left = whole[:c_size, :c_size]
        top_right = whole[:c_size, r_size:]
        bottom_left = whole[r_size:, :c_size]
        bottom_right = whole[r_size:, r_size:]

        # corners.append(rotate_random(top_left))
        # corners.append(rotate_random(top_right))
        # corners.append(rotate_random(bottom_left))
        # corners.append(rotate_random(bottom_right))
        corners.append(top_left)
        corners.append(top_right)
        corners.append(bottom_left)
        corners.append(bottom_right)

        # for i, c in enumerate(corners):
        #     cv2.imwrite('test' + str(i) + '.png', (1 - c) * 255)
    all_imgs.extend(corners)

    print('Loaded ' + str(len(all_imgs)) + ' images')

    all_imgs_f = []
    for i, img in enumerate(all_imgs):
        if i % 40 == 0:
            all_imgs_f.append(img)
    save_imgs(all_imgs_f, 'sketches')

    with open('./sketchrnn_corpus.txt', 'wb') as fp:
        to_save = all_imgs
        pickle.dump(to_save, fp)
