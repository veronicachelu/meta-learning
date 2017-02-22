import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from utils import normalized_columns_initializer

FLAGS = tf.app.flags.FLAGS


class GACNetwork:
    def __init__(self, nb_actions):
        self.nb_actions = nb_actions

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device('gpu:0'):
                self.global_step = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
                self.increment_global_step = self.global_step.assign_add(1)
                self.inputs = tf.placeholder(
                    shape=[None, FLAGS.resized_height, FLAGS.resized_width, FLAGS.agent_history_length],
                    dtype=tf.float32,
                    name="Input")
                self.rewards = tf.placeholder(tf.float32, [None], name='Reward')
                self.actions_onehot = tf.placeholder(tf.float32, [None, self.nb_actions])

                conv1 = tf.contrib.layers.conv2d(
                    self.inputs, 16, 5, 2, activation_fn=tf.nn.relu, scope="conv1")
                conv2 = tf.contrib.layers.conv2d(
                    conv1, 32, 5, 2, padding="VALID", activation_fn=tf.nn.relu, scope="conv2")

                # Fully connected layer
                hidden = tf.contrib.layers.fully_connected(
                    inputs=tf.contrib.layers.flatten(conv2),
                    num_outputs=32,
                    scope="fc1")

                self.value = tf.squeeze(tf.contrib.layers.fully_connected(
                    inputs=hidden,
                    num_outputs=1,
                    activation_fn=None, scope="value"), axis=[1])

                self.policy = tf.contrib.layers.fully_connected(hidden, self.nb_actions, activation_fn=None,
                                                                scope="policy")
                self.policy = tf.nn.softmax(self.policy, name="policy") + 1e-8

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = tf.reduce_sum(tf.square(self.rewards - self.value))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(
                    tf.log(self.responsible_outputs) * (self.rewards - tf.stop_gradient(self.value)))

                self.loss = FLAGS.beta_v * self.value_loss + self.policy_loss - self.entropy * FLAGS.beta_e

                self.optimizer = tf.train.RMSPropOptimizer(FLAGS.lr, 0.99, 0.0, 0.1)
                self.gradients = self.optimizer.compute_gradients(self.loss)
                grads, self.grad_norms = tf.clip_by_average_norm(self.gradients, FLAGS.gradient_clip_value)
                self.grad_clipped = [(tf.clip_by_average_norm(g, FLAGS.gradient_clip_value), v) for g, v in
                                     self.gradients]
                self.apply_grads = self.optimizer.apply_gradients(self.grad_clipped)

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                saver = tf.train.Saver(max_to_keep=5)

                if FLAGS.resume:
                    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
                    print("Loading Model from {}".format(ckpt.model_checkpoint_path))
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                else:
                    self.sess.run(tf.global_variables_initializer())

    def increment_global_step(self):
        self.sess.run(self.increment_global_step)

    def predict(self, s):
        feed_dict = {self.inputs: s}

        pi, v = self.sess.run(
            [self.policy, self.value],
            feed_dict=feed_dict)
        return pi, v



