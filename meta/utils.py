import numpy as np
import random
import tensorflow as tf
import scipy.misc
from scipy.signal import lfilter
import os
import csv
import itertools
import tensorflow.contrib.slim as slim
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from math import floor


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


# This code allows gifs to be saved of the training episode for use in the Control Center.
def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration, verbose=False)


def set_image_bandit(values, probs, selection, trial):
    bandit_image = Image.open('./resources/bandit.png')
    draw = ImageDraw.Draw(bandit_image)
    font = ImageFont.truetype("./resources/FreeSans.ttf", 24)
    draw.text((40, 10), str(float("{0:.2f}".format(probs[0]))), (0, 0, 0), font=font)
    draw.text((130, 10), str(float("{0:.2f}".format(probs[1]))), (0, 0, 0), font=font)
    draw.text((60, 370), 'Trial: ' + str(trial), (0, 0, 0), font=font)
    bandit_image = np.array(bandit_image)
    bandit_image[115:115 + floor(values[0] * 2.5), 20:75, :] = [0, 255.0, 0]
    bandit_image[115:115 + floor(values[1] * 2.5), 120:175, :] = [0, 255.0, 0]
    bandit_image[101:107, 10 + (selection * 95):10 + (selection * 95) + 80, :] = [80.0, 80.0, 225.0]
    return bandit_image


def set_image_bandit_11_arms(values, target_arm, selection, trial):
    bandit_image = Image.open('./resources/11arm.png')
    draw = ImageDraw.Draw(bandit_image)
    font = ImageFont.truetype("./resources/FreeSans.ttf", 24)
    print("target arm is {}. Selection is {}".format(target_arm, selection))
    delta = 90
    # for i in range(11):
        # if target_arm == i:
    draw.text((40 + 1 * delta, 10), "T", (0, 0, 0), font=font)
        # elif i == 10:
    draw.text((40 + 2 * delta, 10), "I {}".format(target_arm), (0, 0, 0), font=font)
        # else:
    draw.text((40 + 0 * delta, 10), "S", (0, 0, 0), font=font)
    draw.text((60, 370), 'Trial: ' + str(trial), (0, 0, 0), font=font)
    bandit_image = np.array(bandit_image)
    delta = 100
    for i in range(11):
        # bandit_image[115:115 + floor(values[0] * 2.5), 20:75, :] = [0, 255.0, 0]
        if i == target_arm:
            bandit_image[115:115 + floor(values[i]/ 5  * 2.5), (20 + delta * 1):(75 + delta * 1), :] = [0, 255.0, 0]
        elif i == 10:
            bandit_image[115:115 + floor(values[i]/ 5 * 2.5), (20 + delta * 2):(75 + delta * 2), :] = [0, 255.0, 0]
        else:
            bandit_image[115:115 + floor(values[i]/ 5 * 2.5), (20 + delta * 0):(75 + delta * 0), :] = [0, 255.0, 0]
    if selection == target_arm:
        bandit_image[101:107, 10 + delta * 1:10 + delta * 1 + 85, :] = [80.0, 80.0, 225.0]
    elif selection == 10:
        bandit_image[101:107, 10 + delta * 2:10 + delta * 2 + 85, :] = [80.0, 80.0, 225.0]
    else:
        bandit_image[101:107, 10 + delta * 0:10 + delta * 0 + 85, :] = [80.0, 80.0, 225.0]
    return bandit_image


def set_image_context(correct, observation, values, selection, trial):
    obs = observation * 225.0
    obs_a = obs[:, 0:1, :]
    obs_b = obs[:, 1:2, :]
    cor = correct * 225.0
    obs_a = scipy.misc.imresize(obs_a, [100, 100], interp='nearest')
    obs_b = scipy.misc.imresize(obs_b, [100, 100], interp='nearest')
    cor = scipy.misc.imresize(cor, [100, 100], interp='nearest')
    bandit_image = Image.open('./resources/c_bandit.png')
    draw = ImageDraw.Draw(bandit_image)
    font = ImageFont.truetype("./resources/FreeSans.ttf", 24)
    draw.text((50, 360), 'Trial: ' + str(trial), (0, 0, 0), font=font)
    draw.text((50, 330), 'Reward: ' + str(values), (0, 0, 0), font=font)
    bandit_image = np.array(bandit_image)
    bandit_image[120:220, 0:100, :] = obs_a
    bandit_image[120:220, 100:200, :] = obs_b
    bandit_image[0:100, 50:150, :] = cor
    bandit_image[291:297, 10 + (selection * 95):10 + (selection * 95) + 80, :] = [80.0, 80.0, 225.0]
    return bandit_image


def set_image_gridworld(frame, color, reward, step):
    a = scipy.misc.imresize(frame, [200, 200], interp='nearest')
    b = np.ones([400, 200, 3]) * 255.0
    b[0:200, 0:200, :] = a
    b[200:210, 0:200, :] = np.array(color) * 255.0
    b = Image.fromarray(b.astype('uint8'))
    draw = ImageDraw.Draw(b)
    font = ImageFont.truetype("./resources/FreeSans.ttf", 24)
    draw.text((40, 280), 'Step: ' + str(step), (0, 0, 0), font=font)
    draw.text((40, 330), 'Reward: ' + str(reward), (0, 0, 0), font=font)
    c = np.array(b)
    return c