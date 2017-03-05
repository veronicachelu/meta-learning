import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from utils import normalized_columns_initializer
from fast_weights import LayerNormFastWeightsBasicRNNCell

FLAGS = tf.app.flags.FLAGS


class ACNetwork():
    def __init__(self, scope, trainer, global_step=None):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, FLAGS.game_size, FLAGS.game_size, FLAGS.game_channels],
                                         dtype=tf.float32, name="Inputs")

            self.image_summaries = []
            with tf.variable_scope('inputs'):
                tf.get_variable_scope().reuse_variables()
                self.image_summaries.append(
                    tf.summary.image('input', self.inputs, max_outputs=1))

            self.conv = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.inputs), 64,
                                                          activation_fn=tf.nn.elu)
            summary_conv_act = tf.contrib.layers.summarize_activation(self.conv)

            self.timestep = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="timestep")

            if FLAGS.meta:
                self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Prev_Rewards")
                self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32, name="Prev_Actions")
                self.prev_actions_onehot = tf.one_hot(self.prev_actions, FLAGS.nb_actions, dtype=tf.float32,
                                                      name="Prev_Actions_OneHot")

                hidden = tf.concat([self.conv, self.prev_rewards, self.prev_actions_onehot,
                                       self.timestep], 1, name="Concatenated_input")
            else:
                hidden = self.conv

            summary_hidden_act = tf.contrib.layers.summarize_activation(hidden)

            rnn_in = tf.expand_dims(hidden, [0], name="RNN_input")
            step_size = tf.shape(self.inputs)[:1]

            if FLAGS.fw:
                rnn_cell = LayerNormFastWeightsBasicRNNCell(48)
                self.initial_state = rnn_cell.zero_state(48)
                self.initial_fast_weights = rnn_cell.zero_fast_weights(48)
                self.state_init = [self.initial_state, self.initial_fast_weights]
                h_in = tf.placeholder(tf.float32, [1, 48], name="hidden_state")
                fw_in = tf.placeholder(tf.float32, [1, 48], name="fast_weights")
                self.state_in = (h_in, fw_in)

                rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
                    rnn_cell, rnn_in, initial_state=self.state_in, sequence_length=step_size,
                    time_major=False)
                rnn_h, rnn_fw = rnn_state
                self.state_out = (rnn_h[:1, :], rnn_fw[:1, :])
                rnn_out = tf.reshape(rnn_outputs, [-1, 48], name="RNN_out")
            else:
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(48, state_is_tuple=True)
                c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
                h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
                self.state_init = [c_init, h_init]
                c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c], name="c_in")
                h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h], name="h_in")
                self.state_in = (c_in, h_in)
                state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

                lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                    lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                    time_major=False)

                lstm_c, lstm_h = lstm_state
                self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
                rnn_out = tf.reshape(lstm_outputs, [-1, 48], name="RNN_out")

            summary_rnn_act = tf.contrib.layers.summarize_activation(rnn_out)

            fc_pol_w = tf.get_variable("FC_Pol_W", shape=[48, FLAGS.nb_actions],
                                       initializer=normalized_columns_initializer(0.01))
            self.policy = tf.nn.softmax(tf.matmul(rnn_out, fc_pol_w, name="Policy"), name="Policy_soft")

            summary_policy_act = tf.contrib.layers.summarize_activation(self.policy)

            fc_value_w = tf.get_variable("FC_Value_W", shape=[48, 1],
                                         initializer=normalized_columns_initializer(1.0))
            self.value = tf.matmul(rnn_out, fc_value_w, name="Value")

            summary_value_act = tf.contrib.layers.summarize_activation(self.value)

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

                self.worker_summaries = [summary_conv_act, summary_hidden_act, summary_rnn_act, summary_policy_act,
                                         summary_value_act]
                for grad, weight in zip(grads, local_vars):
                    self.worker_summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
                    self.worker_summaries.append(tf.summary.histogram(weight.name, weight))

                self.merged_summary = tf.summary.merge(self.worker_summaries)

                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

