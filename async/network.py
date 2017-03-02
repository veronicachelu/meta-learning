import math
from math import sqrt

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


class ACNetwork:
    def __init__(self, scope, nb_actions, trainer):
        with tf.variable_scope(scope):

            self.inputs = tf.placeholder(
                shape=[None, FLAGS.resized_height, FLAGS.resized_width, FLAGS.agent_history_length], dtype=tf.float32,
                name="Input")

            self.image_summaries = []
            with tf.variable_scope('inputs'):
                tf.get_variable_scope().reuse_variables()
                for i in range(FLAGS.agent_history_length):
                    self.image_summaries.append(
                        tf.summary.image('inputs/frames/{}'.format(i), tf.expand_dims(self.inputs[:, :, :, i], axis=3),
                                         max_outputs=1))

            fan_in = 4 * FLAGS.conv1_kernel_size * FLAGS.conv1_kernel_size
            fan_out = FLAGS.conv1_kernel_size * FLAGS.conv1_kernel_size * FLAGS.conv1_nb_kernels
            w_bound = np.sqrt(6. / (fan_in + fan_out))

            conv1 = tf.contrib.layers.conv2d(
                self.inputs, FLAGS.conv1_nb_kernels, FLAGS.conv1_kernel_size, FLAGS.conv1_stride,
                activation_fn=tf.nn.elu, padding=FLAGS.conv1_padding,
                weights_initializer=tf.random_uniform_initializer(-w_bound, w_bound),
                biases_initializer=tf.constant_initializer(0.0),
                variables_collections=tf.get_collection("variables_" + scope),
                outputs_collections=("activations_" + scope),
                scope="conv1")

            fan_in = FLAGS.conv1_nb_kernels * FLAGS.conv2_kernel_size * FLAGS.conv2_kernel_size
            fan_out = FLAGS.conv2_kernel_size * FLAGS.conv2_kernel_size * FLAGS.conv2_nb_kernels
            w_bound = np.sqrt(6. / (fan_in + fan_out))

            conv2 = tf.contrib.layers.conv2d(
                conv1, FLAGS.conv2_nb_kernels, FLAGS.conv2_kernel_size, FLAGS.conv2_stride, padding=FLAGS.conv2_padding,
                weights_initializer=tf.random_uniform_initializer(-w_bound, w_bound),
                biases_initializer=tf.constant_initializer(0.0),
                variables_collections=tf.get_collection("variables_" + scope),
                outputs_collections=("activations_" + scope),
                activation_fn=tf.nn.elu, scope="conv2")

            conv2_flatten = tf.contrib.layers.flatten(conv2)

            fan_in = conv2_flatten.get_shape().as_list()[1]
            fan_out = FLAGS.fc_size
            w_bound = np.sqrt(6. / (fan_in + fan_out))

            hidden = tf.contrib.layers.fully_connected(
                inputs=conv2_flatten,
                num_outputs=FLAGS.fc_size,
                activation_fn=tf.nn.elu,
                weights_initializer=tf.random_uniform_initializer(-w_bound, w_bound),
                biases_initializer=tf.constant_initializer(0.0),
                variables_collections=tf.get_collection("variables_" + scope),
                outputs_collections=("activations_" + scope),
                scope="fc1")

            with tf.variable_scope('conv1'):
                tf.get_variable_scope().reuse_variables()
                weights = tf.get_variable('weights')
                grid = self.put_kernels_on_grid(weights)
                for i in range(FLAGS.agent_history_length):
                    self.image_summaries.append(
                        tf.summary.image('conv1/features/{}'.format(i), tf.expand_dims(grid[:, :, :, i], axis=3),
                                         max_outputs=1))
                for i in range(FLAGS.conv1_nb_kernels):
                    self.image_summaries.append(
                        tf.summary.image('conv1/activation/{}'.format(i), tf.expand_dims(conv1[:, :, :, i], axis=3),
                                         max_outputs=1))

            self.policy = tf.contrib.layers.fully_connected(hidden, nb_actions, activation_fn=None,
                                                            weights_initializer=self.normalized_columns_initializer(0.01),
                                                            biases_initializer=None,
                                                            variables_collections=tf.get_collection(
                                                                "variables_" + scope),
                                                            outputs_collections=("activations_" + scope),
                                                            activation_fn=tf.nn.softmax,
                                                            scope="policy_fc")
            self.policy = tf.add(self.policy, tf.constant(1e-30), name='policy')

            self.value = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=self.normalized_columns_initializer(1.0),
                biases_initializer=None,
                variables_collections=tf.get_collection("variables_" + scope),
                outputs_collections=("activations_" + scope),
                scope="value")

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
                self.value_loss = tf.reduce_sum(tf.nn.l2_loss(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = FLAGS.beta_v * self.value_loss + self.policy_loss - self.entropy * FLAGS.beta_e

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, FLAGS.gradient_clip_value)

                self.worker_summaries = []
                self.worker_summaries.append(
                    tf.contrib.layers.summarize_collection("variables_" + scope, summarizer=tf.contrib.layers.summarize_tensor))
                self.worker_summaries.append(tf.contrib.layers.summarize_collection("activations_" + scope,
                                                                             summarizer=tf.contrib.layers.summarize_activation))

                for grad, weight in zip(grads, local_vars):
                    self.worker_summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
                    # self.worker_summaries.append(tf.summary.histogram(weight.name, weight))

                self.merged_summary = tf.summary.merge(self.worker_summaries)

                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars),
                                                           global_step=tf.contrib.framework.get_global_step())

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

    # Used to initialize weights for policy and value output layers
    def normalized_columns_initializer(self, std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)

        return _initializer

    def xavier_std(self, in_size, out_size):
        return np.sqrt(2. / (in_size + out_size))