import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from utils import normalized_columns_initializer

FLAGS = tf.app.flags.FLAGS


class ACNetwork:
    def __init__(self, scope, nb_actions, trainer):
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


            self.policy = tf.contrib.layers.fully_connected(hidden, nb_actions, activation_fn=None, scope="policy")
            self.policy = tf.nn.softmax(self.policy, name="policy") + 1e-8

            summary_policy_act = tf.contrib.layers.summarize_activation(self.policy)

            self.value = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=1,
                activation_fn=None, scope="value")

            summary_value_act = tf.contrib.layers.summarize_activation(self.value)

            if scope != 'global':

                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, nb_actions, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.max_value = tf.reduce_max(tf.reshape(self.value, [-1]))
                self.min_value = tf.reduce_min(tf.reshape(self.value, [-1]))
                self.mean_value = tf.reduce_mean(tf.reshape(self.value, [-1]))

                self.max_reward = tf.reduce_max(self.target_v)
                self.min_reward = tf.reduce_min(self.target_v)
                self.mean_reward = tf.reduce_mean(self.target_v)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = FLAGS.beta_v * self.value_loss + self.policy_loss - self.entropy * FLAGS.beta_e

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, FLAGS.gradient_clip_value)

                self.worker_summaries = [summary_conv1_act, summary_conv2_act, summary_linear_act, summary_policy_act,
                                         summary_value_act]
                for grad, weight in zip(self.gradients, local_vars):
                    self.worker_summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
                    self.worker_summaries.append(tf.summary.histogram(weight.name, weight))

                self.merged_summary = tf.summary.merge(self.worker_summaries)

                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


