import os
from flask import Flask, request, render_template, jsonify
import base64
import numpy as np
import sys
import cv2
from PIL import Image, ImageDraw
import re
import io
import time
import random
import math

from similar import Repeater, align_images_safe

repeater = Repeater()

np.set_printoptions(threshold=np.inf)

# Support for gomix's 'front-end' and 'back-end' UI.
app = Flask(__name__, static_folder='public', template_folder='views')

def save_img(img, name):
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)
    cv2.imwrite(name, img)


@app.route('/')
def homepage():
    '''Displays the homepage.'''
    return render_template('index.html')


def get_img_array(data):
    meta, data_url = data.split(',')
    img_bytes = io.BytesIO(base64.b64decode(data_url))
    img = Image.open(img_bytes)
    img_arr = np.array(img)[:, :, 0]
    return img_arr


def get_data_url(img_arr):
    img_f = (1 - img_arr.squeeze()) * 255
    img_f = img_f.astype(np.uint8)
    pil_img = Image.fromarray(img_f)
    buff = io.BytesIO()
    pil_img.save(buff, format='PNG')
    img_data_url = base64.b64encode(buff.getvalue()).decode('utf-8')
    return img_data_url


def smoothLineSkeleton(img):
    kernel = np.ones((2, 2), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    # first we widen it slighlty
    skel = cv2.dilate(img, kernel, iterations=1)
    # save_img(skel, 'after_adjust1_dil.png')

    # then we use a median blur to smooth pixelation
    skel = cv2.medianBlur(skel, 5)
    # save_img(skel, 'after_adjust2_medblur.png')

    # then we erode it to make it narrow again
    skel = cv2.erode(skel, kernel, iterations=1)
    # save_img(skel, 'after_adjust3_erode.png')

    # then we blur it to make it have smooth dissipation
    skel = cv2.blur(skel, (2, 2))
    # save_img(skel, 'after_adjust4_smoothedge.png')

    return skel


def transformLineWeight(img_orig, scaleFactor=1, max_iterations=100):
    kernel = np.ones((2, 2), np.uint8)
    # Based on http://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html
    img = img_orig.copy()
    img = img * 255
    img = img.astype(np.uint8)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # ret, img = cv2.threshold(img, 0.5, 1, 0)
    # ret, img = cv2.threshold(img, 90, 255, cv2.THRESH_TOZERO)

    # cv2.imwrite('after_adjust00_init.png', img)

    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imwrite('after_adjust01_init_norm.png', img)

    iterations = int(math.ceil(scaleFactor) - 1)
    if iterations > 0:
        img = cv2.erode(img, kernel, iterations=iterations)
    # cv2.imwrite('after_adjust02_init_erode.png', img)

    ret, img = cv2.threshold(img, 127, 255, 0)
    # cv2.imwrite('after_adjust03_init_thresh.png', img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    iterations = 0

    while (not done and iterations < max_iterations):
        eroded = cv2.erode(img, element)
        # save_img(eroded, 'tr1' + str(iterations) + '.png')
        # cv2.imwrite('tr' + str(iterations) + '1.png', eroded)
        temp = cv2.dilate(eroded, element)
        # save_img(temp, 'tr2' + str(iterations) + '.png')
        # cv2.imwrite('tr' + str(iterations) + '2.png', temp)
        temp = cv2.subtract(img, temp)
        # save_img(temp, 'tr3' + str(iterations) + '.png')
        # cv2.imwrite('tr' + str(iterations) + '3.png', temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        # cv2.imwrite('after_adjust04' + str(iterations) + '.png', skel)

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
        iterations += 1

    # we have a 1px wide skeleton, but it is pixelated, we want to make look like a mark
    # save_img(skel, 'after_adjust10_skel.png')
    # # first we widen it slighlty
    # skel = cv2.dilate(skel, kernel, iterations=1)
    # save_img(skel, 'after_adjust11_dil.png')

    # skel = smoothLineSkeleton(skel)
    # # save_img(skel, 'after_adjust1_1.png')
    # skel = smoothLineSkeleton(skel)
    # # save_img(skel, 'after_adjust1_2.png')
    # skel = smoothLineSkeleton(skel)
    # # save_img(skel, 'after_adjust1_3.png')
    # # skel = smoothLineSkeleton(skel)
    # # save_img(skel, 'after_adjust_4.png')

    # # then we erode it to make it narrow again
    # iterations = int(math.ceil(scaleFactor))
    # skel = cv2.erode(skel, kernel, iterations=iterations)
    # # save_img(skel, 'after_adjust12_erode_' + str(iterations) + '.png')

    # # kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # skel = cv2.filter2D(skel, -1, kernel_sharpen)
    # # save_img(skel, 'after_adjust13_sharpen.png')

    # skel = cv2.blur(skel, (2, 2))
    # save_img(skel, 'after_adjust14_blur.png')

    skel = skel / 255.0

    return skel


def format(img):
    img = img / 255.0
    img = 1 - img
    h, w = img.shape
    img = np.asarray(img).astype(np.float32).reshape((h, w, 1))
    return img


@app.route('/actions', methods=['GET'])
def actions():
    start_time = time.time()
    befores = request.args.getlist('befores[]')
    mark = request.args.get('mark')
    afters = request.args.getlist('afters[]')
    imgs = request.args.getlist('imgs[]')
    # bounds = request.args.getlist('bounds[]')
    # bounds = [float(i) for i in bounds]
    # markBounds = request.args.getlist('markBounds[]')
    # markBounds = [float(i) for i in markBounds]
    n = int(request.args.get('n'))

    befores_f = map(get_img_array, befores[:3])
    afters_f = map(get_img_array, afters[:4])
    imgs_f = map(get_img_array, imgs[:3])
    mark_to_match = get_img_array(mark)
    print('Mark to match shape', mark_to_match.shape)

    # save_img(befores_f[0], 'before' + str(0) + '.png')

    # find pieces of images that are similar
    start_time_before = time.time()
    befores_matches = repeater.get_similar_befores(imgs_f, befores_f, n * 4)

    before_matches_L4 = []
    img_L3 = format(imgs_f[2])
    extra_matches = []
    if len(befores_matches[2]) > 2:
        extra_matches = befores_matches[2][2:]
        befores_matches[2] = befores_matches[2][:2]
        print('extra!')
    else:
        extra_matches = befores_matches[2]

    for before_match in extra_matches:
        before_match_img, location = before_match
        x, y = location['x'] - 12, location['y'] - 12
        end_x, end_y = x + 105, y + 105
        location = {'x': x, 'y': y}
        before_match_img = img_L3[x:end_x, y:end_y]
        h, w, c = before_match_img.shape
        before_match_img = cv2.copyMakeBorder(before_match_img, 0, (105 - h), 0, (105 - w), cv2.BORDER_CONSTANT, value=[0., 0., 0.])
        before_matches_L4.append((before_match_img, location))
    befores_matches.append(before_matches_L4)

    print('Fetched before matches in --- %s seconds ---' % (time.time() - start_time_before))

    # Get matches for marks at L2 and L4 (with a min and max threshold to select 'marks')
    start_time_marks = time.time()
    mark_matches4 = repeater.get_similar_marks(mark_to_match, 4, 3)
    mark_matches2 = repeater.get_similar_marks(mark_to_match, 2, 3)
    mark_matches = []
    mark_matches.extend(mark_matches2)
    mark_matches.extend(mark_matches4)
    print('Fetched marks in --- %s seconds ---' % (time.time() - start_time_marks))

    # For each layer
    start_time_bestmarks = time.time()
    best_option_by_layer = []
    for i, before_matches in enumerate(befores_matches):
        options = []
        for before_match in before_matches:
            before_match_img, location = before_match
            # Try some random selection of the marks on before
            marks = []
            if i < 2:
                marks = random.sample(mark_matches2, 2)
            else:
                marks = random.sample(mark_matches4, 2)

            for mark in marks:
                testAfter = before_match_img.copy().squeeze()
                # print(testAfter.shape, mark.shape, mark_to_match.shape)

                # save_img(mark, 'adjust_before.png')
                # we want to resize mark to match largest original mark dim
                dim_to_match = max(mark_to_match.shape[0], mark_to_match.shape[1])
                mark = cv2.resize(mark, (dim_to_match, dim_to_match), interpolation=cv2.INTER_AREA)

                # then we want to dilate or erode to make line weights equal
                # save_img(mark, '2before_adjust.png')
                # scaleFactor = dim_to_match / 45  # 45 is the size of L2
                # scaleFactor = dim_to_match / 81  # 81 is the size of L3
                # scaleFactor = dim_to_match / 105  # 105 is the size of L4
                # mark = transformLineWeight(mark, scaleFactor)

                if i < 2:
                    # mark = cv2.normalize(mark, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    kernel = np.ones((2, 2), np.uint8)
                    mark = cv2.erode(mark, kernel, iterations=1)
                    ret, mark = cv2.threshold(mark, 0.4, 1, 0)
                    # save_img(mark, 'adjust_after.png')

                # Pad outside of mark
                h_pad = before_match_img.shape[1] - mark.shape[1]
                h_pad_left = int(h_pad / 2)
                h_pad_right = h_pad - h_pad_left
                v_pad = before_match_img.shape[0] - mark.shape[0]
                v_pad_top = int(v_pad / 2)
                v_pad_bottom = v_pad - v_pad_top
                print(before_match_img.shape, mark.shape, v_pad_top, v_pad_bottom, h_pad_left, h_pad_right)
                if h_pad >= 0 or v_pad >= 0:
                    mark = cv2.copyMakeBorder(mark, v_pad_top, v_pad_bottom, h_pad_left, h_pad_right, cv2.BORDER_CONSTANT, value=[0., 0., 0.])
                else:
                    mark = cv2.resize(mark, before_match_img.shape[:2], interpolation=cv2.INTER_AREA)

                # Add mark to before
                testAfter += mark
                options.append((testAfter, mark, before_match_img, location))

        # Get best option for this before by comparing to after
        options = repeater.sort_options(afters_f[i], options, i)
        best_option = options[0]
        best_option_by_layer.append(best_option)
    print('Fetched best marks in --- %s seconds ---' % (time.time() - start_time_bestmarks))

    # encode the images
    result = {}
    for i in range(len(befores_matches)):
        best_option = best_option_by_layer[i]
        befores = [before_match[0] for before_match in befores_matches[i]]
        result[i] = {
            'location': best_option[3],
            'before': get_data_url(best_option[2]),
            'mark': get_data_url(best_option[1]),
            'after': get_data_url(best_option[0]),
            'marks': map(get_data_url, mark_matches),
            'befores': map(get_data_url, befores)
        }


    print('Served suggestions in --- %s seconds ---' % (time.time() - start_time))

    return jsonify(result)


if __name__ == '__main__':
    app.debug = True
    # app.run(
    #     host=os.getenv('LISTEN', '0.0.0.0'),
    #     port=int(os.getenv('PORT', '8080'))
    # )
    app.run()
