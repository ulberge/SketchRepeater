# SketchRepeat

![gif of interaction](http://www.erikulberg.com/imgs/gifs/sketchrepeat1.gif)

## Description

Open-ended sketching helps to explore or create new ideas. While AI systems such as [sketch-rnn](https://magenta.tensorflow.org/assets/sketch_rnn_demo/index.html) and [ShadowDraw](http://vision.cs.utexas.edu/projects/shadowdraw/shadowdraw.html) can support sketching with clear goals, their prediction and auto-completion are limited to classes in their datasets. Open-ended sketching remains difficult to support because a goal is required to narrow the infinite search space of suggestions.

This project presents an interactive drawing tool that provides suggestions by extrapolating the user's previous action. It makes the strong assumption that the user will want to semantically repeat their actions. The goal of the tool is to "play along" with the designer as they move towards an undefined goal. The hope is that the designer finds the AI suggestions useful and inspiring.

**Technologies:** TensorFlow, Spotify ANNOY, Python, JS, p5.js

## How it Works

When a user makes a mark, the system records the mark, the state of the canvas at its location before the mark, and the state after the mark. The before state is matched to other areas in the current canvas and the mark is matched to a corpus of chopped up human sketches (from the [TU Berlin dataset](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/) and the [Google "Quick, Draw" dataset](https://quickdraw.withgoogle.com/data)). "Matches" are judged by comparing the distance between activations from intermediate layers of [Sketch-A-Net](http://sketchx.eecs.qmul.ac.uk/downloads/), a convolutional neural network. The matches are used to generate suggestions which are compared to the after state. The best suggestions are sent back and displayed.

Below is a debug view to demonstrate functionality. The user draws in the top center. AI suggestions appear in the four canvases below. If the user selects an AI mark suggestion, it is added as the next mark and new suggestions are fetched.

![gif of interaction 2](http://www.erikulberg.com/imgs/gifs/sketchrepeat2.gif)

In the following example, a user has just added a circle on top of the triangle. The “before”, mark, and “after” images can be seen on the right. The AI suggestions are displayed below along with their “before”, mark, and “after” images. The AIs take the mark and the location of the match and use an agent-based algorithm to draw something similar to the mark in a matching style.

![how much did the AI contribute?](http://www.erikulberg.com/imgs/stills/ballontri.png)

## How well does it work?

The AI suggestions are most relevant in images of patterns or repetitive landscapes. In the examples below, the human marks are in orange and AI marks are in blue.

![how much did the AI contribute 1?](http://www.erikulberg.com/imgs/stills/sketchrepeat_examples1.png)
![how much did the AI contribute 2?](http://www.erikulberg.com/imgs/stills/sketchrepeat_examples2.png)

## Running the code

**Before setup:**
- Download the pre-trained weights for Sketch-A-Net and the TU Berlin dataset [here](http://sketchx.eecs.qmul.ac.uk/downloads/) (the program uses dataset_without_order_info_224.mat and model_without_order_info_224.mat)
- Download the ndjson files for [Google "Quick, Draw"](https://quickdraw.withgoogle.com/data) for circles, clouds, hexagos, lines, octagons, pillows, squares, squiggles, triangles, and zigzags. 
- Put all these files in the data/ folder.

**Setup:**
cd data
python sketch_process.py

**Run server:**
python server.py

In order for this to run well, a lot of tuning of various parameters has to happen.
