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

    cv2.imwrite('after_adjust00_init.png', img)

    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite('after_adjust01_init_norm.png', img)

    iterations = int(math.ceil(scaleFactor) - 1)
    if iterations > 0:
        img = cv2.erode(img, kernel, iterations=iterations)
    cv2.imwrite('after_adjust02_init_erode.png', img)

    ret, img = cv2.threshold(img, 127, 255, 0)
    cv2.imwrite('after_adjust03_init_thresh.png', img)
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
        cv2.imwrite('after_adjust04' + str(iterations) + '.png', skel)

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
        iterations += 1

    # we have a 1px wide skeleton, but it is pixelated, we want to make look like a mark
    save_img(skel, 'after_adjust10_skel.png')
    # first we widen it slighlty
    skel = cv2.dilate(skel, kernel, iterations=1)
    save_img(skel, 'after_adjust11_dil.png')

    skel = smoothLineSkeleton(skel)
    save_img(skel, 'after_adjust1_1.png')
    skel = smoothLineSkeleton(skel)
    save_img(skel, 'after_adjust1_2.png')
    skel = smoothLineSkeleton(skel)
    save_img(skel, 'after_adjust1_3.png')
    # skel = smoothLineSkeleton(skel)
    # save_img(skel, 'after_adjust_4.png')

    # then we erode it to make it narrow again
    iterations = int(math.ceil(scaleFactor))
    skel = cv2.erode(skel, kernel, iterations=iterations)
    save_img(skel, 'after_adjust12_erode_' + str(iterations) + '.png')

    # kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    skel = cv2.filter2D(skel, -1, kernel_sharpen)
    save_img(skel, 'after_adjust13_sharpen.png')

    skel = cv2.blur(skel, (2, 2))
    save_img(skel, 'after_adjust14_blur.png')

    # # first we widen it slighlty
    # skel = cv2.dilate(skel, kernel, iterations=2)
    # save_img(skel, 'after_adjust1_dil.png')

    # # then we use a median blur to smooth pixelation
    # skel = cv2.medianBlur(skel, 5)
    # save_img(skel, 'after_adjust2_medblur.png')

    # # then we erode it to make it narrow again
    # skel = cv2.erode(skel, kernel)
    # save_img(skel, 'after_adjust3_erode.png')

    # # then we blur it to make it have smooth dissipation
    # skel = cv2.blur(skel, (2, 2))
    # save_img(skel, 'after_adjust4_smoothedge.png')


    # skel = cv2.blur(skel, (3, 3))
    # save_img(skel, 'after_adjust1_blur.png')
    # kernel = np.ones((2, 2), np.uint8)
    # skel = cv2.erode(skel, kernel, iterations=1)
    # save_img(skel, 'after_adjust2_erode.png')
    # skel = cv2.medianBlur(skel, 5)
    # save_img(skel, 'after_adjust3_medblur.png')
    # save_img(cv2.GaussianBlur(skel, (5, 5), 0), 'after_adjust2_test4.png')
    # save_img(cv2.GaussianBlur(skel, (3, 3), 0), 'after_adjust2_test5.png')
    # save_img(cv2.erode(skel, kernel, iterations=1), 'after_adjust2_test6.png')
    # save_img(cv2.erode(skel, kernel, iterations=2), 'after_adjust2_test7.png')
    # skel = cv2.dilate(skel, kernel, iterations=1)
    # skel = cv2.erode(skel, kernel, iterations=1)
    # save_img(skel, 'after_adjust2_erode2.png')
    # skel = cv2.dilate(skel, kernel, iterations=1)
    # skel = cv2.erode(skel, kernel, iterations=1)
    # save_img(skel, 'after_adjust2_erode3.png')
    # img = img_orig.copy()
    # img = img * 255
    # img = img.astype(np.uint8)

    # ret, thresh = cv2.threshold(skel, 127, 255, 0)
    # cv2.imwrite('after_adjust3_thresh.png', thresh)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # cv2.imwrite('after_tr_contimg.png', cont_img)
    # cont_test = np.zeros(skel.shape)
    # cont_test = cv2.drawContours(cont_test, contours, 0, (255), 1)
    # cv2.imwrite('after_adjust4_conts.png', cont_test)

    skel = skel / 255.0

    return skel


def transformLineWeight2(img_orig, scaleFactor=1, max_iterations=100):
    kernel = np.ones((2, 2), np.uint8)
    img = img_orig.copy()
    img = img * 255
    img = img.astype(np.uint8)

    cv2.imwrite('after_adjust00_init.png', img)

    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite('after_adjust01_init_norm.png', img)

    # iterations = int(math.ceil(scaleFactor) + 1)
    iterations = int(math.ceil(scaleFactor) - 1)
    if iterations > 0:
        img = cv2.erode(img, kernel, iterations=iterations)
    cv2.imwrite('after_adjust02_init_erode.png', img)

    skel = img

    # kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    skel = cv2.filter2D(skel, -1, kernel_sharpen)
    save_img(skel, 'after_adjust03_sharpen.png')

    # first we widen it slighlty
    # skel = cv2.dilate(skel, kernel, iterations=1)
    # save_img(skel, 'after_adjust11_dil.png')

    skel = smoothLineSkeleton(skel)
    save_img(skel, 'after_adjust1_1.png')
    skel = smoothLineSkeleton(skel)
    save_img(skel, 'after_adjust1_2.png')
    skel = smoothLineSkeleton(skel)
    save_img(skel, 'after_adjust1_3.png')
    # skel = smoothLineSkeleton(skel)
    # save_img(skel, 'after_adjust_4.png')

    # then we erode it to make it narrow again
    if iterations > 0:
        skel = cv2.erode(skel, kernel, iterations=iterations)
    save_img(skel, 'after_adjust22_erode.png')

    # kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    skel = cv2.filter2D(skel, -1, kernel_sharpen)
    save_img(skel, 'after_adjust23_sharpen.png')

    # remvoe blobs
    blobs = cv2.erode(skel, kernel, iterations=3)
    blobs = cv2.dilate(blobs, kernel, iterations=2)
    cv2.imwrite('after_adjust24_blobs.png', blobs)
    ret, blob_filter = cv2.threshold(blobs, 50, 255, 0)
    cv2.imwrite('after_adjust25_blob_filter.png', blob_filter)
    skel = cv2.subtract(skel, blob_filter)
    cv2.imwrite('after_adjust26_noblob.png', skel)


    # skel = cv2.filter2D(skel, -1, kernel_sharpen)
    # save_img(skel, 'after_adjust23_sharpen2.png')

    # skel = cv2.blur(skel, (2, 2))
    # save_img(skel, 'after_adjust24_blur.png')



    skel = skel / 255.0

    return skel


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
    afters_f = map(get_img_array, afters)
    imgs_f = map(get_img_array, imgs)
    mark_to_match = get_img_array(mark)
    print('Mark to match shape', mark_to_match.shape)

    print('bounds', bounds, bounds[3] - bounds[1], bounds[2] - bounds[0])
    print('mark', mark_to_match.shape)
    for before in befores_f:
        print('before', before.shape)
    for after in afters_f:
        print('after', after.shape)

    # save_img(befores_f[0], 'before' + str(0) + '.png')

    # find pieces of images that are similar
    befores_matches = repeater.get_similar_befores(imgs_f, befores_f, n)

    for i, befores in enumerate(befores_matches):
        for before in befores:
            print('before match L' + str(i + 1), before[0].shape)

    # Get matches for marks at L2 (with a min and max threshold to select 'marks')
    mark_matches = repeater.get_similar_marks(mark_to_match, 9)

    for mark in mark_matches:
        print('mark match', mark.shape)

    # For each layer
    actions_by_layer = []
    for i, before_matches in enumerate(befores_matches):
        before_match_imgs = [match[0] for match in before_matches]
        after = afters_f[i]
        actions = []
        for before_match_img in before_match_imgs:
            # Try some random selection of the marks on before
            marks = random.sample(mark_matches, 5)
            options = []
            for mark in marks:
                before = before_match_img.copy().squeeze()
                print(before.shape, mark.shape, mark_to_match.shape)

                save_img(mark, 'before_resize.png')
                # we want to resize mark to match largest original mark dim
                dim_to_match = max(mark_to_match.shape[0], mark_to_match.shape[1])
                mark = cv2.resize(mark, (dim_to_match, dim_to_match), interpolation=cv2.INTER_AREA)
                print(mark.shape)

                # then we want to crop to mark mark match original mark
                save_img(mark, 'before_crop.png')
                if mark_to_match.shape[0] < dim_to_match:
                    diff = dim_to_match - mark_to_match.shape[0]
                    half_diff0 = diff / 2
                    half_diff1 = diff - half_diff0
                    mark = mark[half_diff0:-half_diff1, :]

                if mark_to_match.shape[1] < dim_to_match:
                    diff = dim_to_match - mark_to_match.shape[1]
                    half_diff0 = diff / 2
                    half_diff1 = diff - half_diff0
                    mark = mark[:, half_diff0:-half_diff1]
                print(mark.shape)
                save_img(mark, 'before_adjust.png')

                # then we want to dilate or erode to make line weights equal
                scaleFactor = dim_to_match / 45
                mark = transformLineWeight2(mark, scaleFactor)
                save_img(mark, 'after_adjust.png')
                print(mark.shape)
                # if dim_to_match > 45:
                #     # we scaled up, so erode since marks white
                #     kernel = np.ones((5, 5), np.uint8)
                #     for k in range(5):
                #         mark = cv2.erode(mark, kernel, iterations=1)
                #         save_img(mark, 'after_adjust' + str(k) + '.png')
                #         mark_test = cv2.blur(mark, (5, 5))
                #         save_img(mark_test, 'after_adjust' + str(k) + '_blur.png')

                # if dim_to_match < 45:
                #     # we scaled down, so dilate since marks white
                #     kernel = np.ones((5, 5), np.uint8)
                #     cv2.dilate(mark, kernel, iterations=1)

                # Pad outside of mark
                print(before.shape, mark.shape)
                h_pad = before.shape[1] - mark.shape[1]
                h_pad_left = int(h_pad / 2)
                h_pad_right = h_pad - h_pad_left

                v_pad = before.shape[0] - mark.shape[0]
                v_pad_top = int(v_pad / 2)
                v_pad_bottom = v_pad - v_pad_top

                print(mark.shape, v_pad_top, v_pad_bottom, h_pad_left, h_pad_right)
                mark = cv2.copyMakeBorder(mark, v_pad_top, v_pad_bottom, h_pad_left, h_pad_right, cv2.BORDER_CONSTANT, value=[0., 0., 0.])

                print(before.shape, mark.shape)
                before += mark
                # action = cv2.normalize(orig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                options.append((before, mark))

            # Get best option for this before by comparing to after
            options = repeater.sort_options(after, options, i)
            best_option = options[0]
            actions.append(best_option[1])

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
