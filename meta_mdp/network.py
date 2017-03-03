import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from utils import normalized_columns_initializer

FLAGS = tf.app.flags.FLAGS


class ACNetwork():
    def __init__(self, scope, trainer, global_step=None):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, FLAGS.game_size, FLAGS.game_size, FLAGS.game_channels],
                                         dtype=tf.float32, name="Inputs")

            self.image_summaries = []
            with tf.variable_scope('inputs'):
                tf.get_variable_scope().reuse_variables()
                for i in range(FLAGS.game_channels):
                    self.image_summaries.append(
                        tf.summary.image('inputs/frames/{}'.format(i), tf.expand_dims(self.inputs[:, :, :, i], axis=3),
                                         max_outputs=1))

            self.conv = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.inputs), 64,
                                                          activation_fn=tf.nn.elu)

            self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Prev_Rewards")
            self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32, name="Prev_Actions")
            self.timestep = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="timestep")
            self.prev_actions_onehot = tf.one_hot(self.prev_actions, FLAGS.nb_actions, dtype=tf.float32,
                                                  name="Prev_Actions_OneHot")

            hidden = tf.concat([tf.contrib.layers.flatten(self.conv), self.prev_rewards, self.prev_actions_onehot,
                                   self.timestep], 1, name="Concatenated_input")

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(48, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c], name="c_in")
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h], name="h_in")
            self.state_in = (c_in, h_in)

            rnn_in = tf.expand_dims(hidden, [0], name="RNN_input")
            step_size = tf.shape(self.prev_rewards)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)

            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 48], name="RNN_out")

            fc_pol_w = tf.get_variable("FC_Pol_W", shape=[48, FLAGS.nb_actions],
                                       initializer=normalized_columns_initializer(0.01))
            self.policy = tf.nn.softmax(tf.matmul(rnn_out, fc_pol_w, name="Policy"), name="Policy_soft")

            fc_value_w = tf.get_variable("FC_Value_W", shape=[48, 1],
                                         initializer=normalized_columns_initializer(1.0))
            self.value = tf.matmul(rnn_out, fc_value_w, name="Value")

            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")
                self.actions_onehot = tf.one_hot(self.actions, FLAGS.nb_actions, dtype=tf.float32,
                                                 name="Actions_Onehot")

                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = FLAGS.beta_v * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))

                starter_beta_e = 1.0
                end_beta_e = 0.0
                decay_steps = 20000
                self.beta_e = tf.train.polynomial_decay(starter_beta_e, global_step,
                                                        decay_steps, end_beta_e,
                                                        power=0.5)

                self.policy_loss = -tf.reduce_sum(
                    tf.log(self.responsible_outputs + 1e-7) * self.advantages) - self.entropy * self.beta_e

                self.loss = self.value_loss + self.policy_loss

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, FLAGS.gradient_clip_value)

                self.worker_summaries = []
                for grad, weight in zip(grads, local_vars):
                    self.worker_summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
                    self.worker_summaries.append(tf.summary.histogram(weight.name, weight))

                self.merged_summary = tf.summary.merge(self.worker_summaries)

                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

