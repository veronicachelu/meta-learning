import tensorflow as tf
import config
import math
import os
import numpy as np

class AsyncDQNetwork:

  def __init__(self, action_size):
    with tf.Graph().as_default():
      self.sess = tf.InteractiveSession()

      self.T = 0

      # network params:
      self.state_input, self.q_values, self.network_params = self.create_network(action_size)

      # target network params:
      self.target_state_input, self.target_q_values, self.target_network_params = self.create_network(action_size)

      # Op for periodically updating target network with online network weights
      self.reset_target_network_params = [self.target_network_params[i].assign(self.network_params[i]) for i in
                                     range(len(self.target_network_params))]

      self.action_input = tf.placeholder("float", [None, action_size])
      self.y_input = tf.placeholder("float", [None])

      self.q_value_output = tf.reduce_sum(tf.mul(self.q_values, self.action_input), reduction_indices=1)

      self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output))
      self.grad_update = tf.train.AdamOptimizer(config.LEARN_RATE).minimize(self.cost, var_list=self.network_params)

      self.saver = tf.train.Saver()

      self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summaries()

      self.sess.run(tf.initialize_all_variables())

      # Initialize target network weights
      self.sess.run(self.reset_target_network_params)

      if not os.path.exists(config.CHECKPOINT_PATH):
        os.mkdir(config.CHECKPOINT_PATH)

      checkpoint = tf.train.get_checkpoint_state(config.CHECKPOINT_PATH)
      if checkpoint and checkpoint.model_checkpoint_path:
        self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
      else:
        print "Could not find old network weights"



      if not os.path.exists(config.SUMMARY_PATH):
        os.mkdir(config.SUMMARY_PATH)

      self.writer = tf.train.SummaryWriter(config.SUMMARY_PATH, self.sess.graph)


  def create_network(self, action_size):
    params = []
    with tf.device("/cpu:0"):
      state_input = tf.placeholder("float", [None, config.RESIZED_SCREEN_X, config.RESIZED_SCREEN_Y,
                                             config.STATE_FRAMES])

      # network weights
      convolution_weights_1 = tf.Variable(tf.truncated_normal([8, 8, config.STATE_FRAMES, 32], stddev=0.01))
      convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[32]))

      params.append(convolution_weights_1)
      params.append(convolution_bias_1)

      hidden_convolutional_layer_1 = tf.nn.relu(
          tf.nn.conv2d(state_input, convolution_weights_1, strides=[1, 4, 4, 1], padding="SAME") + convolution_bias_1)

      hidden_max_pooling_layer_1 = tf.nn.max_pool(hidden_convolutional_layer_1, ksize=[1, 2, 2, 1],
                                                  strides=[1, 2, 2, 1], padding="SAME")

      convolution_weights_2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
      convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[64]))

      params.append(convolution_weights_2)
      params.append(convolution_bias_2)

      hidden_convolutional_layer_2 = tf.nn.relu(
          tf.nn.conv2d(hidden_max_pooling_layer_1, convolution_weights_2, strides=[1, 2, 2, 1],
                       padding="SAME") + convolution_bias_2)

      hidden_max_pooling_layer_2 = tf.nn.max_pool(hidden_convolutional_layer_2, ksize=[1, 2, 2, 1],
                                                  strides=[1, 2, 2, 1], padding="SAME")

      convolution_weights_3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
      convolution_bias_3 = tf.Variable(tf.constant(0.01, shape=[64]))

      params.append(convolution_weights_3)
      params.append(convolution_bias_3)

      hidden_convolutional_layer_3 = tf.nn.relu(
          tf.nn.conv2d(hidden_max_pooling_layer_2, convolution_weights_3,
                       strides=[1, 1, 1, 1], padding="SAME") + convolution_bias_3)

      hidden_max_pooling_layer_3 = tf.nn.max_pool(hidden_convolutional_layer_3, ksize=[1, 2, 2, 1],
                                                  strides=[1, 2, 2, 1], padding="SAME")

      hidden_max_pooling_layer_3_shape = hidden_max_pooling_layer_3.get_shape()[1] * \
                                         hidden_max_pooling_layer_3.get_shape()[2] * \
                                         hidden_max_pooling_layer_3.get_shape()[3]
      hidden_max_pooling_layer_3_shape = hidden_max_pooling_layer_3_shape.value

      hidden_convolutional_layer_3_flat = tf.reshape(hidden_max_pooling_layer_3, [-1, hidden_max_pooling_layer_3_shape])

      feed_forward_weights_1 = tf.Variable(tf.truncated_normal([hidden_max_pooling_layer_3_shape, 256], stddev=0.01))
      feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))

      params.append(feed_forward_weights_1)
      params.append(feed_forward_bias_1)

      final_hidden_activations = tf.nn.relu(
          tf.matmul(hidden_convolutional_layer_3_flat, feed_forward_weights_1) + feed_forward_bias_1)

      feed_forward_weights_2 = tf.Variable(tf.truncated_normal([256, action_size], stddev=0.01))
      feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[action_size]))

      params.append(feed_forward_weights_2)
      params.append(feed_forward_bias_2)

      output_layer = tf.matmul(final_hidden_activations, feed_forward_weights_2) + feed_forward_bias_2

      return state_input, output_layer, params


  # def save_network(self, time_step):
  #   self.saver.save(self.sess, config.CHECKPOINT_PATH, global_step=time_step)
  #
  #
  # def get_action(self, last_state):
  #   # choose an action given our last state
  #   target_q_values = self.sess.run(self.q_values, feed_dict={self.state_input: [last_state]})[0]
  #   # if self.verbose_logging:
  #   #   print("Action Q-Values are %s" % readout_t)
  #   action = np.argmax(target_q_values)
  #   return action
  #
  #
  # def get_target_q_batch(self, state_batch):
  #   Q_s_a = self.sess.run(self.q_values, feed_dict={self.state_input: state_batch})
  #   return Q_s_a
  #
  #
  # def train(self, y_batch, state_batch, one_hot_actions):
  #   # learn that these actions in these states lead to this reward
  #   self.sess.run(self.train_operation, feed_dict={
  #     self.state_input: state_batch,
  #     self.action_input: one_hot_actions,
  #     self.y_input: y_batch})
  #


  def get_action(self, last_state):
    # choose an action given our last state
    q_values = self.sess.run(self.q_values, feed_dict={self.state_input: [last_state]})[0]
    # if self.verbose_logging:
    #   print("Action Q-Values are %s" % readout_t)
    action = np.argmax(q_values)
    return action


  def get_q_values(self, last_state):
    q_values = self.sess.run(self.q_values, feed_dict={self.state_input: [last_state]})[0]
    return q_values


  def get_target_q_values(self, state_batch):
    Q_s_a = self.sess.run(self.target_q_values, feed_dict={self.target_state_input: [state_batch]})[0]
    return Q_s_a


  def reset_target_network(self):
    self.sess.run(self.reset_target_network_params)


  def run_grad_update(self, y_batch, state_batch, one_hot_actions):
    # learn that these actions in these states lead to this reward
    self.sess.run(self.grad_update, feed_dict={
      self.state_input: state_batch,
      self.action_input: one_hot_actions,
      self.y_input: y_batch})


  def save_network(self, time_step):
    self.saver.save(self.sess, config.CHECKPOINT_PATH, global_step=time_step)


  def setup_summaries(self):
    episode_reward = tf.Variable(0.)
    tf.scalar_summary("Episode Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.scalar_summary("Max Q Value", episode_ave_max_q)
    logged_epsilon = tf.Variable(0.)
    tf.scalar_summary("Epsilon", logged_epsilon)
    summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
    summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.merge_all_summaries()
    return summary_placeholders, update_ops, summary_op


  def update_summaries(self, stats):
    for i in range(len(stats)):
      self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(stats[i])})


  def run_summary_op(self):
    return self.sess.run(self.summary_op)