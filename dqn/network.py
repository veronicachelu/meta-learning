import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
FLAGS = tf.app.flags.FLAGS

class DQNetwork:
    def __init__(self, nb_actions, scope):
        with tf.variable_scope(scope):

            self.inputs = tf.placeholder(
                shape=[None, FLAGS.resized_height, FLAGS.resized_width, FLAGS.agent_history_length], dtype=tf.float32,
                name="Input")

            conv1 = tf.contrib.layers.conv2d(
                self.inputs, 16, 5, 2, activation_fn=tf.nn.relu, scope="conv1")
            conv2 = tf.contrib.layers.conv2d(
                conv1, 32, 5, 2, padding="VALID", activation_fn=tf.nn.relu, scope="conv2")

            hidden = tf.contrib.layers.fully_connected(
                inputs=tf.contrib.layers.flatten(conv2),
                num_outputs=32,
                scope="fc1")

            summary_conv1_act = tf.contrib.layers.summarize_activation(conv1)
            summary_conv2_act = tf.contrib.layers.summarize_activation(conv2)
            summary_linear_act = tf.contrib.layers.summarize_activation(hidden)

            self.action_values = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=nb_actions,
                activation_fn=None, scope="value")

            summary_action_value_act = tf.contrib.layers.summarize_activation(self.action_values)

            if scope != 'target':

                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, nb_actions, dtype=tf.float32)
                self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)

                self.action_value = tf.reduce_sum(tf.multiply(self.action_values, self.actions_onehot), reduction_indices=1)

                # Loss functions
                self.action_value_loss = tf.reduce_sum(tf.square(self.target_q - self.action_value))

                self.optimizer = tf.train.AdamOptimizer(FLAGS.lr)
                grads = tf.gradients(self.action_value_loss, tf.trainable_variables())
                grads = list(zip(grads, tf.trainable_variables()))

                self.train_operation = self.optimizer.apply_gradients(grads)

                self.summaries = [summary_conv1_act, summary_conv2_act, summary_linear_act, summary_action_value_act]

                for grad, weight in grads:
                    self.summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
                    self.summaries.append(tf.summary.histogram(weight.name, weight))

                self.merged_summary = tf.summary.merge(self.summaries)


