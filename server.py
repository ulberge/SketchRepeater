from flask import Flask, request, render_template, jsonify
from repeater import Repeater
import time
from helpers import get_data_url, get_img_arr, get_np_arr

# Get instance of Repeater which can fetch and judge matches
repeater = Repeater()

# Support for gomix's 'front-end' and 'back-end' UI.
app = Flask(__name__, static_folder='public', template_folder='views')


@app.route('/')
def homepage():
    '''Displays the main page.'''
    return render_template('index.html')


@app.route('/actions', methods=['GET'])
def actions():
    '''API for getting suggested actions'''
    start_time = time.time()

    # The image sections before the mark at sizes appropriate to CNN layer
    befores = request.args.getlist('befores[]')
    # The mark made
    mark = request.args.get('mark')
    # The image sections after the mark at sizes appropriate to CNN layer
    afters = request.args.getlist('afters[]')
    # The current state of each AI canvas
    imgs = request.args.getlist('imgs[]')

    # format the dataURLs into image arrays
    befores_f = map(get_np_arr, map(get_img_arr, befores))
    mark_f = get_np_arr(get_img_arr(mark))
    afters_f = map(get_np_arr, map(get_img_arr, afters))
    imgs_f = map(get_np_arr, map(get_img_arr, imgs))

    best_actions_by_layer = repeater.get_suggested_actions(befores_f, mark_f, afters_f, imgs_f)

    # format from image arrays to dataURL
    result = {}
    for i, action in enumerate(best_actions_by_layer):
        if action is not None:
            result[i] = {
                'location': action[3],
                'before': get_data_url(action[2]),
                'mark': get_data_url(action[1]),
                'after': get_data_url(action[0])
            }

    print('Served suggestions in --- %s seconds ---' %
          (time.time() - start_time))

    return jsonify(result)


if __name__ == '__main__':
    app.debug = True
    app.run()
