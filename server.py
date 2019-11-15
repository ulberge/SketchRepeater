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


@app.route('/suggestions', methods=['GET'])
def suggestions():
    start_time = time.time()
    befores = request.args.getlist('befores[]')
    afters = request.args.getlist('afters[]')
    img = request.args.get('img')
    bounds = request.args.getlist('bounds[]')
    bounds = [float(i) for i in bounds]
    n = int(request.args.get('n'))

    befores_f = map(get_img_array, befores)
    afters_f = map(get_img_array, afters)
    img_f = get_img_array(img)

    # save_img(befores_f[0], 'before' + str(0) + '.png')

    # find pieces of images that are similar
    befores_matches = repeater.get_similar_before(img_f, befores_f, bounds, n)

    # for i, before in enumerate(befores_matches[0]):
    #     save_img(before, 'before_matches' + str(i) + '.png')

    # afters_matches = repeater.get_similar_after(afters_f, n)

    # align images
    # afters_matches_aligned = []
    # for i, before_matches in enumerate(befores_matches):
    #     after_matches = afters_matches[i]
    #     after_matches_aligned = []
    #     for j, before in enumerate(before_matches):
    #         after = after_matches[j]
    #         after_aligned = align_images_safe(before, after)
    #         after_matches_aligned.append(after_aligned)
    #     afters_matches_aligned.append(after_matches_aligned)

    # encode the images
    result = {}
    for i in range(len(befores_matches)):
        result[i] = {
            'locations': [match[1] for match in before_matches[i]],
            'locationImgs': map(get_data_url, [match[0] for match in before_matches[i]]),
            # 'suggestions': map(get_data_url, afters_matches[i]),
            # 'suggestionsAligned': map(get_data_url, afters_matches_aligned[i])
        }


    print('Served suggestions in --- %s seconds ---' % (time.time() - start_time))

    return jsonify(result)


@app.route('/actions', methods=['GET'])
def actions():
    start_time = time.time()
    befores = request.args.getlist('befores[]')
    mark = request.args.get('mark')
    afters = request.args.getlist('afters[]')
    imgs = request.args.getlist('imgs[]')
    bounds = request.args.getlist('bounds[]')
    bounds = [float(i) for i in bounds]
    n = int(request.args.get('n'))

    befores_f = map(get_img_array, befores)
    # marks_f = map(get_img_array, marks)
    afters_f = map(get_img_array, afters)
    imgs_f = map(get_img_array, imgs)
    mark_to_match = get_img_array(mark)
    print('Mark to match shape', mark_to_match.shape)

    # save_img(befores_f[0], 'before' + str(0) + '.png')

    # find pieces of images that are similar
    befores_matches = repeater.get_similar_befores(imgs_f, befores_f, n)

    # Get matches for marks at L2 (with a min and max threshold to select 'marks')
    mark_matches = repeater.get_similar_marks(mark_to_match, n * 5)

    # For each layer
    actions_by_layer = []
    for i, before_matches in enumerate(befores_matches):
        before_match_imgs = [match[0] for match in before_matches]
        actions = []
        for before_match_img in before_match_imgs:
            # Try some random selection of the marks on before
            orig = np.array(before_match_img.copy().squeeze())
            mark = random.sample(mark_matches, 1)[0]
            mark = cv2.resize(mark, (mark_to_match.shape[1], mark_to_match.shape[0]), interpolation=cv2.INTER_AREA)
            # pad
            h_pad_left = int(math.ceil((orig.shape[1] - mark.shape[1]) / 2.0))
            h_pad_right = orig.shape[1] - mark.shape[1] - h_pad_left
            v_pad_top = int(math.ceil((orig.shape[0] - mark.shape[0]) / 2.0))
            v_pad_bottom = orig.shape[0] - mark.shape[0] - v_pad_top
            mark = cv2.copyMakeBorder(mark, v_pad_top, v_pad_bottom, h_pad_left, h_pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            mark = cv2.resize(mark, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_AREA)
            orig += mark
            action = orig
            # action = cv2.normalize(orig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            actions.append(action)

        # Sort combos based on diff to after for this layer

        actions_by_layer.append(actions)


    # for i, before in enumerate(befores_matches[0]):
    #     save_img(before, 'before_matches' + str(i) + '.png')

    # afters_matches = repeater.get_similar_after(afters_f, n)

    # align images
    # afters_matches_aligned = []
    # for i, before_matches in enumerate(befores_matches):
    #     after_matches = afters_matches[i]
    #     after_matches_aligned = []
    #     for j, before in enumerate(before_matches):
    #         after = after_matches[j]
    #         after_aligned = align_images_safe(before, after)
    #         after_matches_aligned.append(after_aligned)
    #     afters_matches_aligned.append(after_matches_aligned)

    # encode the images
    result = {}
    for i in range(len(befores_matches)):
        result[i] = {
            'locations': [match[1] for match in befores_matches[i]],
            'locationImgs': map(get_data_url, [match[0] for match in befores_matches[i]]),
            'marks': map(get_data_url, mark_matches),
            'actions': map(get_data_url, actions_by_layer[i])
            # 'suggestionsAligned': map(get_data_url, afters_matches_aligned[i])
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
