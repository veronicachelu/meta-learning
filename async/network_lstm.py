import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from utils import normalized_columns_initializer

FLAGS = tf.app.flags.FLAGS


class ACNetworkLSTM:
    def __init__(self, scope, nb_actions, trainer):
        with tf.variable_scope(scope):
            if FLAGS.meta:
                self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Prev_Rewards")
                self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32, name="Prev_Actions")
                self.prev_actions_onehot = tf.one_hot(self.prev_actions, nb_actions, dtype=tf.float32,
                                                      name="Prev_Actions_OneHot")

            self.inputs = tf.placeholder(
                shape=[None, FLAGS.resized_height, FLAGS.resized_width, FLAGS.agent_history_length], dtype=tf.float32,
                name="Input")

            conv1 = tf.contrib.layers.conv2d(
                self.inputs, 32, 5, 2, activation_fn=tf.nn.relu, scope="conv1")
            conv2 = tf.contrib.layers.conv2d(
                conv1, 32, 5, 2, padding="VALID", activation_fn=tf.nn.relu, scope="conv2")

            # Fully connected layer
            hidden = tf.contrib.layers.fully_connected(
                inputs=tf.contrib.layers.flatten(conv2),
                num_outputs=64,
                scope="fc1")

            if FLAGS.meta:
                hidden = tf.concat([self.prev_rewards, self.prev_actions_onehot, hidden], 1,
                                   name="Concatenated_input")

            summary_conv1_act = tf.contrib.layers.summarize_activation(conv1)
            summary_conv2_act = tf.contrib.layers.summarize_activation(conv2)
            summary_linear_act = tf.contrib.layers.summarize_activation(hidden)

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(32, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c], name="c_in")
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h], name="h_in")
            self.state_in = (c_in, h_in)

            rnn_in = tf.expand_dims(hidden, [0], name="RNN_input")
            step_size = tf.shape(self.inputs)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])

            rnn_out = tf.reshape(lstm_outputs, [-1, 32], name="RNN_out")

            self.policy = tf.contrib.layers.fully_connected(rnn_out, nb_actions, activation_fn=None, scope="policy")
            self.policy = tf.nn.softmax(self.policy, name="policy") + 1e-8

            summary_policy_act = tf.contrib.layers.summarize_activation(self.policy)

            self.value = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=1,
                activation_fn=None, scope="value")

            summary_value_act = tf.contrib.layers.summarize_activation(self.value)

            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")
                self.actions_onehot = tf.one_hot(self.actions, nb_actions, dtype=tf.float32, name="Actions_Onehot")
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
                for grad, weight in zip(grads, local_vars):
                    self.worker_summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
                    self.worker_summaries.append(tf.summary.histogram(weight.name, weight))

                self.merged_summary = tf.summary.merge(self.worker_summaries)

                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


    def xavier_initializer(self, uniform=True, seed=None, dtype=dtypes.float32):
        return self.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                 uniform=uniform, seed=seed, dtype=dtype)

    def variance_scaling_initializer(self, factor=2.0, mode='FAN_IN', uniform=False,
                                     seed=None, dtype=dtypes.float32):
        if not dtype.is_floating:
            raise TypeError('Cannot create initializer for non-floating point type.')
        if mode not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG']:
            raise TypeError('Unknow mode %s [FAN_IN, FAN_OUT, FAN_AVG]', mode)

        # pylint: disable=unused-argument
        def _initializer(shape, dtype=dtype, partition_info=None):
            """Initializer function."""
            if not dtype.is_floating:
                raise TypeError('Cannot create initializer for non-floating point type.')
            # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
            # This is the right thing for matrix multiply and convolutions.
            if shape:
                fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
                fan_out = float(shape[-1])
            else:
                fan_in = 1.0
                fan_out = 1.0
            for dim in shape[:-2]:
                fan_in *= float(dim)
                fan_out *= float(dim)
            if mode == 'FAN_IN':
                # Count only number of input connections.
                n = fan_in
            elif mode == 'FAN_OUT':
                # Count only number of output connections.
                n = fan_out
            elif mode == 'FAN_AVG':
                # Average number of inputs and output connections.
                n = (fan_in + fan_out) / 2.0
            if uniform:
                # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
                limit = math.sqrt(3.0 * factor / n)
                return random_ops.random_uniform(shape, -limit, limit,
                                                 dtype, seed=seed)
            else:
                # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
                trunc_stddev = math.sqrt(1.3 * factor / n)
                return random_ops.truncated_normal(shape, 0.0, trunc_stddev, dtype,
                                                   seed=seed)

        # pylint: enable=unused-argument

        return _initializer
