import tensorflow as tf
import flags
import math
import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, GRU, Bidirectional, Embedding, TimeDistributed, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import time

FLAGS = tf.app.flags.FLAGS


class AsyncAC3Network:
    def __init__(self, action_size):
        with tf.Graph().as_default():
            self.sess = tf.InteractiveSession()

            self.T = 0
            # Store layers weight & bias
            self.weights = {
                'conv1_w': tf.get_variable("Conv1_W", shape=[8, 8, FLAGS.STATE_FRAMES, 32],
                                           initializer=tf.contrib.layers.xavier_initializer()),
                'conv2_w': tf.get_variable("Conv2_W", shape=[4, 4, 32, 64],
                                           initializer=tf.contrib.layers.xavier_initializer()),
                'conv3_w': tf.get_variable("Conv3_W", shape=[3, 3, 64, 64],
                                           initializer=tf.contrib.layers.xavier_initializer()),
                'fc1_w': tf.get_variable("FC1_W", shape=[64, 256],
                                         initializer=tf.contrib.layers.xavier_initializer()),
                'fc2_w': tf.get_variable("FC2_pol_W", shape=[256, action_size],
                                         initializer=tf.contrib.layers.xavier_initializer()),
                'fc3_w': tf.get_variable("FC3_val_W", shape=[256, 1],
                                         initializer=tf.contrib.layers.xavier_initializer()),
            }

            self.biases = {
                'conv1_b': tf.Variable(tf.constant(0.01, shape=[32]), name='Conv1_b'),
                'conv2_b': tf.Variable(tf.constant(0.01, shape=[64]), name="Conv2_b"),
                'conv3_b': tf.Variable(tf.constant(0.01, shape=[64]), name="Conv3_b"),
                'fc1_b': tf.Variable(tf.constant(0.01, shape=[256]), name="FC1_b"),
                'fc2_b': tf.Variable(tf.constant(0.01, shape=[action_size]), name="FC2_pol_b"),
                'fc3_b': tf.Variable(tf.constant(0.01, shape=[1]), name="FC3_val_b")
            }

            with tf.name_scope('Model'):
                # network params:
                self.state_input, self.policy_output, self.value_output, self.policy_params, self.value_params, \
                self.lstm_state_out, self.initial_lstm_state0, self.initial_lstm_state1, self.step_size, self.lstm_state = \
                    self.create_network(action_size)

            self.action_input = tf.placeholder("float", [None, action_size])
            self.r_input = tf.placeholder("float", [None])

            with tf.name_scope('Loss'):
                # policy entropy
                self.entropy = -tf.reduce_sum(self.policy_output * tf.log(self.policy_output), reduction_indices=1)
                self.loss_policy = -tf.reduce_sum(tf.reduce_sum(tf.mul(tf.log(self.policy_output), self.action_input),
                                                                reduction_indices=1) * (
                                                      self.r_input - self.value_output) +
                                                  self.entropy * FLAGS.ENTROPY_BETA)
                tf.scalar_summary('Policy loss (raw)', self.loss_policy)
                loss_policy_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
                self.loss_policy_averages_op = loss_policy_averages.apply([self.loss_policy])
                tf.scalar_summary('Policy loss (avg)', loss_policy_averages.average(self.loss_policy))

                self.loss_value = tf.reduce_mean(tf.square(self.r_input - self.value_output))
                tf.scalar_summary('Value loss (raw)', self.loss_policy)
                loss_value_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
                self.loss_value_averages_op = loss_value_averages.apply([self.loss_value])
                tf.scalar_summary('Value loss (avg)', loss_value_averages.average(self.loss_value))

                self.total_loss = self.loss_policy + (0.5 * self.loss_value)
                tf.scalar_summary('Total loss (raw)', self.loss_policy)
                loss_total_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
                self.loss_total_averages_op = loss_total_averages.apply([self.total_loss])
                tf.scalar_summary('Total loss (avg)', loss_total_averages.average(self.total_loss))

            with tf.name_scope('SGD'):
                self.optimizer = tf.train.AdamOptimizer(FLAGS.LEARN_RATE)
                grads = tf.gradients(self.total_loss, tf.trainable_variables())
                grads = list(zip(grads, tf.trainable_variables()))
                # Op to update all variables according to their gradient
                self.train_op = self.optimizer.apply_gradients(grads_and_vars=grads)

            # Create summaries to visualize weights
            for var in tf.trainable_variables():
                tf.histogram_summary(var.name, var)
            # Summarize all gradients
            for grad, var in grads:
                tf.histogram_summary(var.name + '/gradient', grad)
            # Merge all summaries into a single op
            # self.merged_summary_op = tf.merge_all_summaries()

            self.saver = tf.train.Saver()

            self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summaries()

            self.sess.run(tf.global_variables_initializer())

            checkpoint = tf.train.get_checkpoint_state(FLAGS.CHECKPOINT_PATH)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                print("Could not find old network weights")

            self.writer = tf.train.SummaryWriter(FLAGS.SUMMARY_PATH, self.sess.graph)

    def create_network_keras(self, action_size):
        with tf.device("/cpu:0"):
            state_input = tf.placeholder("float", [None, config.RESIZED_SCREEN_X, config.RESIZED_SCREEN_Y,
                                                   config.STATE_FRAMES])

            inputs = Input(shape=(config.RESIZED_SCREEN_X, config.RESIZED_SCREEN_Y, config.STATE_FRAMES))
            shared = Convolution2D(name="conv1", nb_filter=16, nb_row=8, nb_col=8, subsample=(4, 4),
                                   activation='relu',
                                   border_mode='same')(inputs)
            shared = Convolution2D(name="conv2", nb_filter=32, nb_row=4, nb_col=4, subsample=(2, 2),
                                   activation='relu',
                                   border_mode='same')(shared)
            shared = Flatten()(shared)
            shared = Dense(name="h1", output_dim=256, activation='relu')(shared)

            action_probs = Dense(name="p", output_dim=action_size, activation='softmax')(shared)

            state_value = Dense(name="v", output_dim=1, activation='linear')(shared)

            policy_network = Model(input=inputs, output=action_probs)
            value_network = Model(input=inputs, output=state_value)

            policy_params = policy_network.trainable_weights
            value_params = value_network.trainable_weights

            policy_output_layer = policy_network(state_input)
            value_output_layer = value_network(state_input)

            return state_input, policy_output_layer, value_output_layer, policy_params, value_params

    def create_network(self, action_size):

        with tf.device("/cpu:0"), tf.variable_scope('net') as scope:
            state_input = tf.placeholder("float", [None, FLAGS.RESIZED_SCREEN_X, FLAGS.RESIZED_SCREEN_Y,
                                                   FLAGS.STATE_FRAMES], name='StateInput')

            conv1 = self.conv2d(state_input, self.weights['conv1_w'], self.biases['conv1_b'], strides=4, padding="SAME")
            relu1 = tf.nn.relu(conv1)
            tf.histogram_summary("conv_relu1", conv1)

            maxpool1 = self.maxpool2d(relu1, k=2, padding="SAME")

            conv2 = self.conv2d(maxpool1, self.weights['conv2_w'], self.biases['conv2_b'], strides=2, padding="SAME")
            relu2 = tf.nn.relu(conv2)
            tf.histogram_summary("conv_relu2", conv2)

            maxpool2 = self.maxpool2d(relu2, k=2, padding="SAME")

            conv3 = self.conv2d(maxpool2, self.weights['conv3_w'], self.biases['conv3_b'], strides=1, padding="SAME")
            relu3 = tf.nn.relu(conv3)
            tf.histogram_summary("conv_relu3", conv3)

            maxpool3 = self.maxpool2d(relu3, k=2, padding="SAME")

            maxpool3_shape = maxpool3.get_shape()[1] * \
                             maxpool3.get_shape()[2] * \
                             maxpool3.get_shape()[3]
            maxpool3_shape = maxpool3_shape.value

            maxpool3_flat = tf.reshape(maxpool3, [-1, maxpool3_shape])

            fc1 = tf.matmul(maxpool3_flat, self.weights['fc1_w']) + self.biases['fc1_b']
            fc1_relu = tf.nn.relu(fc1)
            tf.histogram_summary("fc_relu1", fc1)

            lstm = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)
            fc1_relu_reshaped = tf.reshape(fc1_relu, [1, -1, 256])
            initial_lstm_state0 = tf.placeholder(tf.float32, [1, 256])
            initial_lstm_state1 = tf.placeholder(tf.float32, [1, 256])
            initial_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(initial_lstm_state0, initial_lstm_state1)
            step_size = tf.placeholder(tf.float32, [1], name="StepSize")

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm, fc1_relu_reshaped,
                                                         initial_state=initial_lstm_state,
                                                         sequence_length=step_size,
                                                         time_major=False)
            lstm_outputs = tf.reshape(lstm_outputs, [-1, 256])
            tf.histogram_summary("lstm", lstm_outputs)

            scope.reuse_variables()
            W_lstm = tf.get_variable("RNN/BasicLSTMCell/Linear/Matrix")
            b_lstm = tf.get_variable("RNN/BasicLSTMCell/Linear/Bias")

            policy_output_layer = tf.nn.softmax(
                tf.matmul(lstm_outputs, self.weights['fc2_w']) + self.biases['fc2_b'])
            value_output_layer = tf.matmul(lstm_outputs, self.weights['fc3_w']) + self.biases['fc3_b']

            lstm_state_out = tf.nn.rnn_cell.LSTMStateTuple(np.zeros([1, 256]),
                                                           np.zeros([1, 256]))

            return (state_input, policy_output_layer, value_output_layer, self.get_policy_params(W_lstm, b_lstm),
                    self.get_value_params(W_lstm, b_lstm), lstm_state_out,
                    initial_lstm_state0, initial_lstm_state1, step_size, lstm_state)

    def get_policy_output(self, last_state):
        [policy_output_values, self.lstm_state_out] = self.sess.run([self.policy_output, self.lstm_state],
                                                                    feed_dict={self.state_input: [last_state],
                                                                               self.initial_lstm_state0:
                                                                                   self.lstm_state_out[0],
                                                                               self.initial_lstm_state1:
                                                                                   self.lstm_state_out[1],
                                                                               self.step_size: [1]})
        return policy_output_values[0]

    def get_value_output(self, last_state):
        [value_output, self.lstm_state_out] = self.sess.run([self.value_output, self.lstm_state], feed_dict={
            self.state_input: [last_state],
            self.initial_lstm_state0: self.lstm_state_out[0],
            self.initial_lstm_state1: self.lstm_state_out[1],
            self.step_size: [1]
        })
        return value_output[0]

    def train(self, state_batch, one_hot_actions, r_input, last_summary_time):
        # learn that these actions in these states lead to this reward
        feed = {
            self.state_input: state_batch,
            self.action_input: one_hot_actions,
            self.r_input: r_input,
            self.initial_lstm_state0:
                self.lstm_state_out[0],
            self.initial_lstm_state1:
                self.lstm_state_out[1],
            self.step_size: [1]
        }
        self.sess.run(self.train_op, feed_dict=feed)

        # write summary statistics

        now = time.time()
        if now - last_summary_time > FLAGS.SUMMARY_INTERVAL:
            summary_str = self.run_summary_op(feed)
            self.writer.add_summary(summary_str, float(self.T))
            last_summary_time = now

        return last_summary_time

    def save_network(self, time_step):
        self.saver.save(self.sess, FLAGS.CHECKPOINT_PATH + '/' + 'checkpoint', global_step=time_step)

    def setup_summaries(self):
        episode_reward = tf.Variable(0.)
        tf.scalar_summary("Episode Reward", episode_reward)

        r_summary_placeholder = tf.placeholder("float")
        update_ep_reward = episode_reward.assign(r_summary_placeholder)

        episode_avg_v = tf.Variable(0.)
        tf.scalar_summary("Episode Value", episode_avg_v)

        v_summary_placeholder = tf.placeholder("float")
        update_ep_val = episode_avg_v.assign(v_summary_placeholder)

        summary_placeholders = [r_summary_placeholder, v_summary_placeholder]
        update_ops = [update_ep_reward, update_ep_val]
        summary_op = tf.merge_all_summaries()
        return summary_placeholders, update_ops, summary_op

    def update_summaries(self, stats):
        for i in range(len(stats)):
            self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(stats[i])})

    def run_summary_op(self, feed):
        return self.sess.run(self.summary_op, feed_dict=feed)

    # Create some wrappers for simplicity
    def conv2d(self, x, W, b, strides=1, padding='SAME'):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2, padding='SAME'):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding=padding)

    def get_policy_params(self, W_lstm, b_lstm):
        policy_params = []
        policy_params.append(self.weights['conv1_w'])
        policy_params.append(self.biases['conv1_b'])
        policy_params.append(self.weights['conv2_w'])
        policy_params.append(self.biases['conv2_b'])
        policy_params.append(self.weights['conv3_w'])
        policy_params.append(self.biases['conv3_b'])
        policy_params.append(self.weights['fc1_w'])
        policy_params.append(self.biases['fc1_b'])
        policy_params.append(W_lstm)
        policy_params.append(b_lstm)
        policy_params.append(self.weights['fc2_w'])
        policy_params.append(self.biases['fc2_b'])
        return policy_params

    def get_value_params(self, W_lstm, b_lstm):
        value_params = []
        value_params.append(self.weights['conv1_w'])
        value_params.append(self.biases['conv1_b'])
        value_params.append(self.weights['conv2_w'])
        value_params.append(self.biases['conv2_b'])
        value_params.append(self.weights['conv3_w'])
        value_params.append(self.biases['conv3_b'])
        value_params.append(self.weights['fc1_w'])
        value_params.append(self.biases['fc1_b'])
        value_params.append(W_lstm)
        value_params.append(b_lstm)
        value_params.append(self.weights['fc3_w'])
        value_params.append(self.biases['fc3_b'])
        return value_params
