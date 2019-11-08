import os
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, initializers

"""
An implementation of Sketch-a-Net in TensorFlow 2 with Keras

Sketch-a-Net: A Deep Neural Network that Beats Humans
https://link-springer-com.proxy.library.cmu.edu/article/10.1007/s11263-016-0932-3

Used this repo for some code and as reference: https://github.com/ayush29feb/Sketch-A-XNORNet/
"""


def load_pretrained(filepath):
    """
    From https://github.com/ayush29feb/Sketch-A-XNORNet/
    Loads the pretrained weights and biases from the pretrained model available
    on http://www.eecs.qmul.ac.uk/~tmh/downloads.html
    Args:
        Takes in the filepath for the pretrained .mat filepath

    Returns:
        Returns the dictionary with all the weights and biases for respective layers
    """
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


def load_model(filepath):
    model = models.Sequential()

    layers = load_layers(filepath)
    for layer in layers:
        model.add(layer)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    return model


def load_layers(filepath):
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

    # L5
    model_layers.append(layers.Conv2D(
        256,
        (3, 3),
        strides=1,
        padding='same',
        kernel_initializer=initializers.Constant(weights['conv5']),
        bias_initializer=initializers.Constant(biases['conv5']),
        activation='relu',
        name='conv5'
    ))
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


def get_avg_pools(activations, pool_size, strides):
    # print('pool_size', pool_size)
    # print('strides', strides)
    # print('acts shape', activations.shape)
    pool_layer = layers.AveragePooling2D(
        pool_size=pool_size,
        strides=strides,
        padding='same'
    )
    pools = pool_layer(activations)
    # print('pools shape', pools.shape)

    return pools


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

L2_concepts = [
    [28],
    [24, 81],
    [23, 30],
    [8],
    [5, 43, 122],
    [124],
    [40],
    [81, 105],
    [32, 81, 103, 113],
    [75, 81],
    [24, 56, 118],
    [8, 13, 81],
    [80],
    [57, 97],
    [31],
    [117],
    [81, 107],
    [67, 78],
    [36],
    [29],
    [104],
    [8, 34],
    [11, 21, 125],
    [24, 111],
    [43],
    [22],
    [38, 89],
    [0, 8],
    [2],
    [72],
    [81, 107],
    [97],
    [48],
    [90, 96],
    [63, 108],
    [11, 111],
    [3, 29, 75, 81],
    [68],
    [24, 29, 81, 118],
    [83],
    [8, 13, 42],
    [6, 30, 67],
    [13, 22, 49, 81],
    [115],
    [81, 121],
    [7],
    [101],
    [36, 91],
    [75],
    [19],
    [79],
    [8, 38],
    [123],
    [24, 61, 111],
    [10],
    [30],
]


if __name__ == '__main__':
    load_model()
