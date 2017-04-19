import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from utils import normalized_columns_initializer
from fast_weights import LayerNormFastWeightsBasicRNNCell
from math import sqrt
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

            self.conv = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.inputs), 64)
            self.conv = tf.contrib.layers.layer_norm(self.conv)
            self.conv = tf.nn.elu(self.conv)

            summary_conv_act = tf.contrib.layers.summarize_activation(self.conv)

            if FLAGS.meta:
                self.timestep = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="timestep")
                # self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Prev_Rewards")
                self.prev_rewards = tf.placeholder(shape=[None], dtype=tf.int32, name="Prev_Rewards")
                self.prev_rewards_onehot = tf.one_hot(self.prev_rewards, 2, dtype=tf.float32,
                                                      name="Prev_Rewards_OneHot")

                self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32, name="Prev_Actions")
                self.prev_actions_onehot = tf.one_hot(self.prev_actions, FLAGS.nb_actions, dtype=tf.float32,
                                                      name="Prev_Actions_OneHot")

                # hidden = tf.concat([self.conv, self.prev_rewards, self.prev_actions_onehot,
                #                        self.timestep], 1, name="Concatenated_input")
                hidden = tf.concat([self.conv, self.prev_rewards_onehot, self.prev_actions_onehot], 1, name="Concatenated_input")
            else:
                hidden = self.conv

            summary_hidden_act = tf.contrib.layers.summarize_activation(hidden)

            rnn_in = tf.expand_dims(hidden, [0], name="RNN_input")
            step_size = tf.shape(self.inputs)[:1]

            if FLAGS.fw:
                rnn_cell = LayerNormFastWeightsBasicRNNCell(48)
                # self.initial_state = rnn_cell.zero_state(tf.shape(self.inputs)[0], tf.float32)
                # self.initial_fast_weights = rnn_cell.zero_fast_weights(tf.shape(self.inputs)[0], tf.float32)
                h_init = np.zeros((1, 48), np.float32)
                fw_init = np.zeros((1, 48, 48), np.float32)
                self.state_init = [h_init, fw_init]
                h_in = tf.placeholder(tf.float32, [1, 48], name="hidden_state")
                fw_in = tf.placeholder(tf.float32, [1, 48, 48], name="fast_weights")
                self.state_in = (h_in, fw_in)

                rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
                    rnn_cell, rnn_in, initial_state=self.state_in, sequence_length=step_size,
                    time_major=False)
                rnn_h, rnn_fw = rnn_state
                self.state_out = (rnn_h[:1, :], rnn_fw[:1, :])
                rnn_out = tf.reshape(rnn_outputs, [-1, 48], name="RNN_out")
            else:
                lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(48)
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

                # starter_beta_e = 1.0
                # end_beta_e = 0.0
                # decay_steps = 20000
                # self.beta_e = tf.train.polynomial_decay(starter_beta_e, global_step,
                #                                         decay_steps, end_beta_e,
                #                                         power=0.5)

                self.policy_loss = -tf.reduce_sum(
                    tf.log(self.responsible_outputs + 1e-7) * self.advantages) - self.entropy * FLAGS.beta_e

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

class ConvNetwork():
    def __init__(self, scope, trainer, global_step=None):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, FLAGS.game_size, FLAGS.game_size, FLAGS.game_channels],
                                         dtype=tf.float32, name="Inputs")

            self.conv = tf.contrib.layers.conv2d(
                self.inputs, 32, 5, 2, activation_fn=tf.nn.elu, scope="conv1")

            self.image_summaries = []
            with tf.variable_scope('conv1'):
                tf.get_variable_scope().reuse_variables()
                weights = tf.get_variable('weights')
                grid = self.put_kernels_on_grid(weights)
                self.image_summaries.append(
                    tf.summary.image('kernels', grid, max_outputs=1))


            with tf.variable_scope('inputs'):
                tf.get_variable_scope().reuse_variables()
                self.image_summaries.append(
                    tf.summary.image('input', self.inputs, max_outputs=1))

            self.fc = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.conv), 64)
            # self.conv = tf.contrib.layers.layer_norm(self.conv)
            self.elu = tf.nn.elu(self.fc)

            summary_conv_act = tf.contrib.layers.summarize_activation(self.elu)

            if FLAGS.meta:
                self.timestep = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="timestep")
                self.prev_rewards = tf.placeholder(shape=[None], dtype=tf.int32, name="Prev_Rewards")
                self.prev_rewards_onehot = tf.one_hot(self.prev_rewards, 2, dtype=tf.float32,
                                                      name="Prev_Rewards_OneHot")
                self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32, name="Prev_Actions")
                self.prev_actions_onehot = tf.one_hot(self.prev_actions, FLAGS.nb_actions, dtype=tf.float32,
                                                      name="Prev_Actions_OneHot")

                if FLAGS.one_hot_reward:
                    hidden = tf.concat([self.elu, self.prev_rewards_onehot, self.prev_actions_onehot], 1, name="Concatenated_input")
                else:
                    hidden = tf.concat([self.elu, self.prev_rewards, self.prev_actions_onehot,
                                        self.timestep], 1, name="Concatenated_input")
            else:
                hidden = self.elu

            summary_hidden_act = tf.contrib.layers.summarize_activation(hidden)

            rnn_in = tf.expand_dims(hidden, [0], name="RNN_input")
            step_size = tf.shape(self.inputs)[:1]

            if FLAGS.fw:
                rnn_cell = LayerNormFastWeightsBasicRNNCell(48)
                # self.initial_state = rnn_cell.zero_state(tf.shape(self.inputs)[0], tf.float32)
                # self.initial_fast_weights = rnn_cell.zero_fast_weights(tf.shape(self.inputs)[0], tf.float32)
                h_init = np.zeros((1, 48), np.float32)
                fw_init = np.zeros((1, 48, 48), np.float32)
                self.state_init = [h_init, fw_init]
                h_in = tf.placeholder(tf.float32, [1, 48], name="hidden_state")
                fw_in = tf.placeholder(tf.float32, [1, 48, 48], name="fast_weights")
                self.state_in = (h_in, fw_in)

                rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
                    rnn_cell, rnn_in, initial_state=self.state_in, sequence_length=step_size,
                    time_major=False)
                rnn_h, rnn_fw = rnn_state
                self.state_out = (rnn_h[:1, :], rnn_fw[:1, :])
                rnn_out = tf.reshape(rnn_outputs, [-1, 48], name="RNN_out")
            else:
                lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(48)
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

                # starter_beta_e = 1.0
                # end_beta_e = 0.0
                # decay_steps = 20000
                # self.beta_e = tf.train.polynomial_decay(starter_beta_e, global_step,
                #                                         decay_steps, end_beta_e,
                #                                         power=0.5)

                self.policy_loss = -tf.reduce_sum(
                    tf.log(self.responsible_outputs + 1e-7) * self.advantages) - self.entropy * FLAGS.beta_e

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

    def put_kernels_on_grid(self, kernel, pad=1):

        '''Visualize conv. features as an image (mostly for the 1st layer).
        Place kernel into a grid, with some paddings between adjacent filters.
        Args:
          kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
          (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                               User is responsible of how to break into two multiples.
          pad:               number of black pixels around each filter (between them)
        Return:
          Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
        '''

        # get shape of the grid. NumKernels == grid_Y * grid_X
        def factorization(n):
            for i in range(int(sqrt(float(n))), 0, -1):
                if n % i == 0:
                    if i == 1: print('Who would enter a prime number of filters')
                    return (i, int(n / i))

        (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)
        print('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)

        kernel1 = (kernel - x_min) / (x_max - x_min)

        # pad X and Y
        x1 = tf.pad(kernel1, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

        # X and Y dimensions, w.r.t. padding
        Y = kernel1.get_shape()[0] + 2 * pad
        X = kernel1.get_shape()[1] + 2 * pad

        channels = kernel1.get_shape()[2]

        # put NumKernels to the 1st dimension
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # organize grid on Y axis
        x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

        # switch X and Y axes
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # organize grid on X axis
        x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

        # back to normal order (not combining with the next step for clarity)
        x6 = tf.transpose(x5, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        # scaling to [0, 255] is not necessary for tensorboard
        return x7
