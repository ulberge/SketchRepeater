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

from similar import Repeater, align_images_safe

# Support for gomix's 'front-end' and 'back-end' UI.
app = Flask(__name__, static_folder='public', template_folder='views')
repeater = Repeater()


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
    n = int(request.args.get('n'))

    befores_f = map(get_img_array, befores)
    afters_f = map(get_img_array, afters)
    img_f = get_img_array(img)

    save_img(befores_f[0], 'before' + str(0) + '.png')

    # find pieces of images that are similar
    befores_matches, before_matches_locations = repeater.get_similar_before(img_f, befores_f, n)

    for i, before in enumerate(befores_matches[0]):
        save_img(before, 'before_matches' + str(i) + '.png')

    afters_matches = repeater.get_similar_after(afters_f, n)

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
            'locations': before_matches_locations[i],
            'locationImgs': map(get_data_url, befores_matches[i]),
            'suggestions': map(get_data_url, afters_matches[i]),
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
