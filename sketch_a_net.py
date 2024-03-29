import os
import scipy.io as sio
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, initializers

'''
An implementation of Sketch-a-Net in TensorFlow 2 with Keras

Sketch-a-Net: A Deep Neural Network that Beats Humans
https://link-springer-com.proxy.library.cmu.edu/article/10.1007/s11263-016-0932-3

Used this repo as reference for Python implementation:
https://github.com/ayush29feb/Sketch-A-XNORNet/
'''


def load_pretrained(filepath):
    '''Loads the pretrained weights and biases from the pretrained model available on http://www.eecs.qmul.ac.uk/~tmh/downloads.html

    Keyword arguments:
        filepath -- the pretrained .mat filepath

    Returns:
        Returns the dictionary with all the weights and biases
    '''
    if filepath is None or not os.path.isfile(filepath):
        print('Pretrained Model Not Available!')
        return None, None

    data = sio.loadmat(filepath)
    weights = {}
    biases = {}
    conv_idxs = [0, 3, 6, 8, 10, 13, 16, 19]
    for i, idx in enumerate(conv_idxs):
        weights['conv' + str(i + 1)] = data['net']['layers'][0][0][0][idx]['filters'][0][0]
        biases['conv' + str(i + 1)] = data['net']['layers'][0][0][0][idx]['biases'][0][0].reshape(-1)

    print('Pretrained Model Loaded!')
    return (weights, biases)


def get_layers_output(layers, layer_names, img):
    '''Get the activations for the layers in layer_names for this img and this
    set of layers

    Keyword arguments:
        layers -- the layers of the CNN to run the image through
        layer_names -- names of the layers to save
        img -- image to run through the CNN

    Returns:
        Returns a list with the resulting activations for the layers in layer_names.
    '''
    # format for tf
    curr = np.array([img, ])
    # execute the layers
    acts = []
    count = 0
    for layer in layers:
        # run layer
        curr = layer(curr)

        # only save matching layers
        if layer.name in layer_names:
            acts.append(curr)
            count += 1

        # cut short if we already matched all the layers we care about
        if count == len(layer_names):
            return acts

    return acts


def load_layers(filepath):
    '''Loads the layers for Sketch-A-Net with pretrained weights and biases

    Keyword arguments:
        filepath -- the pretrained .mat filepath

    Returns:
        Returns a list of TF layers for Sketch-A-Net
    '''
    weights, biases = load_pretrained(filepath)

    model_layers = []

    # L1
    model_layers.append(layers.Conv2D(
        64,
        (15, 15),
        strides=3,
        padding='valid',
        kernel_initializer=initializers.Constant(weights['conv1']),
        bias_initializer=initializers.Constant(biases['conv1']),
        activation='relu',
        input_shape=(None, None, 1),
        name='conv1'
    ))
    model_layers.append(layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='valid',
        name='Pool1'
    ))

    # L2
    model_layers.append(layers.Conv2D(
        128,
        (5, 5),
        strides=1,
        padding='valid',
        kernel_initializer=initializers.Constant(weights['conv2']),
        bias_initializer=initializers.Constant(biases['conv2']),
        activation='relu',
        name='conv2'
    ))
    model_layers.append(layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='valid',
        name='Pool2'
    ))

    # L3
    model_layers.append(layers.Conv2D(
        256,
        (3, 3),
        strides=1,
        padding='same',
        kernel_initializer=initializers.Constant(weights['conv3']),
        bias_initializer=initializers.Constant(biases['conv3']),
        activation='relu',
        name='conv3'
    ))

    # L4
    model_layers.append(layers.Conv2D(
        256,
        (3, 3),
        strides=1,
        padding='same',
        kernel_initializer=initializers.Constant(weights['conv4']),
        bias_initializer=initializers.Constant(biases['conv4']),
        activation='relu',
        name='conv4'
    ))

    # Do not need any of the layers past 4 for this project

    # # L5
    # model_layers.append(layers.Conv2D(
    #     256,
    #     (3, 3),
    #     strides=1,
    #     padding='same',
    #     kernel_initializer=initializers.Constant(weights['conv5']),
    #     bias_initializer=initializers.Constant(biases['conv5']),
    #     activation='relu',
    #     name='conv5'
    # ))
    # model_layers.append(layers.MaxPooling2D(
    #     pool_size=(3, 3),
    #     strides=2,
    #     padding='valid',
    #     name='Pool5'
    # ))

    # # L6
    # model_layers.append(layers.Conv2D(
    #     512,
    #     (7, 7),
    #     strides=1,
    #     padding='valid',
    #     kernel_initializer=initializers.Constant(weights['conv6']),
    #     bias_initializer=initializers.Constant(biases['conv6']),
    #     activation='relu',
    #     name='conv6'
    # ))
    # model_layers.append(layers.Dropout(
    #     rate=0.5,
    #     name='Dropout8'
    # ))

    # # L7
    # model_layers.append(layers.Conv2D(
    #     512,
    #     (1, 1),
    #     strides=1,
    #     padding='valid',
    #     kernel_initializer=initializers.Constant(weights['conv7']),
    #     bias_initializer=initializers.Constant(biases['conv7']),
    #     activation='relu',
    #     name='conv7'
    # ))
    # model_layers.append(layers.Dropout(
    #     rate=0.5,
    #     name='Dropout7'
    # ))

    # # L8
    # model_layers.append(layers.Conv2D(
    #     250,
    #     (1, 1),
    #     strides=1,
    #     padding='valid',
    #     kernel_initializer=initializers.Constant(weights['conv8']),
    #     bias_initializer=initializers.Constant(biases['conv8']),
    #     activation='relu',
    #     name='conv8'
    # ))
    # model_layers.append(layers.Dense(250, activation='softmax'))

    return model_layers

# Not necessary for this project
# def load_model(filepath):
#     '''Creates a TF model for Sketch-A-Net from pre-trained weights
#     Keyword arguments:
#         filepath -- the pretrained .mat filepath

#     Returns:
#         Returns the TF model
#     '''
#     model = models.Sequential()

#     layers = load_layers(filepath)
#     for layer in layers:
#         model.add(layer)

#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     model.summary()

#     return model


layers_meta = [
    [
        # params for L1
        'conv1',  # layer_name
        # 71,  # output_size
        3,  # stride
        15,  # f_size
        0  # padding
    ],
    [
        # params for L2
        'conv2',  # layer_name
        # 31,  # output_size
        6,  # stride
        45,  # f_size
        0  # padding
    ],
    [
        # params for L3
        'conv3',  # layer_name
        # 15,  # output_size
        12,  # stride
        81,  # f_size
        12  # padding
    ],
    [
        # params for L4
        'conv4',  # layer_name
        # 15,  # output_size
        12,  # stride
        105,  # f_size
        24  # padding
    ],
    [
        # params for L5
        'conv5',  # layer_name
        # 15,  # output_size
        12,  # stride
        129,  # f_size
        36  # padding
    ]
]

if __name__ == '__main__':
    load_model()
