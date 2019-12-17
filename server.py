from flask import Flask, request, render_template, jsonify
from sketch_repeat import Repeater

# Get instance of Repeater which can fetch and judge matches
repeater = Repeater()

# Support for gomix's 'front-end' and 'back-end' UI.
app = Flask(__name__, static_folder='public', template_folder='views')


@app.route('/')
def homepage():
    '''Displays the homepage.'''
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
    # The number of suggested actions to fetch
    n = int(request.args.get('n'))

    results = repeater.get_suggested_actions(befores, mark, afters, imgs, n)

    print('Served suggestions in --- %s seconds ---' %
          (time.time() - start_time))

    return jsonify(result)


if __name__ == '__main__':
    app.debug = True
    app.run()
