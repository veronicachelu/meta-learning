import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from utils import normalized_columns_initializer
from math import sqrt

FLAGS = tf.app.flags.FLAGS


class FUNNetwork():
    def __init__(self, scope, trainer, global_step=None):
        with tf.variable_scope(scope):
            self.prob_of_random_goal = tf.Variable(FLAGS.initial_random_goal_prob, trainable=False,
                                                   name="prob_of_random_goal", dtype=tf.float32)
            self.inputs = tf.placeholder(shape=[None, FLAGS.game_size, FLAGS.game_size, FLAGS.game_channels],
                                         dtype=tf.float32, name="Inputs")

            self.prev_rewards = tf.placeholder(shape=[None], dtype=tf.int32, name="Prev_Rewards")
            self.prev_rewards_onehot = tf.one_hot(self.prev_rewards, 2, dtype=tf.float32,
                                                  name="Prev_Rewards_OneHot")
            self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32, name="Prev_Actions")
            self.prev_actions_onehot = tf.one_hot(self.prev_actions, FLAGS.nb_actions, dtype=tf.float32,
                                                  name="Prev_Actions_OneHot")
            self.image_summaries = []
            if FLAGS.game_size > 5:
                self.conv = tf.contrib.layers.conv2d(
                    self.inputs, 32, 5, 2, activation_fn=tf.nn.elu, scope="conv1")
                with tf.variable_scope('conv1'):
                    tf.get_variable_scope().reuse_variables()
                    weights = tf.get_variable('weights')
                    grid = self.put_kernels_on_grid(weights)
                    self.image_summaries.append(
                        tf.summary.image('kernels', grid, max_outputs=1))
            else:
                self.conv = self.inputs

            with tf.variable_scope('inputs'):
                tf.get_variable_scope().reuse_variables()
                self.image_summaries.append(
                    tf.summary.image('input', self.inputs, max_outputs=100))

            self.fc = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.conv), FLAGS.hidden_dim)
            self.fc = tf.contrib.layers.layer_norm(self.fc)
            self.f_percept = tf.nn.elu(self.fc, name="Zt")

            self.f_percept = tf.concat([self.f_percept, self.prev_rewards_onehot], 1,
                                       name="Zt_r")

            summary_f_percept_act = tf.contrib.layers.summarize_activation(self.f_percept)

            ############################################################################################################
            # Manager network

            self.f_Mspace = tf.contrib.layers.fully_connected(self.f_percept, FLAGS.hidden_dim)
            self.f_Mspace = tf.contrib.layers.layer_norm(self.f_Mspace)
            self.f_Mspace = tf.nn.elu(self.f_Mspace, name="St")
            summary_f_Mspace_act = tf.contrib.layers.summarize_activation(self.f_Mspace)

            m_rnn_in = tf.expand_dims(self.f_Mspace, [0], name="Mrnn_in")
            step_size = tf.shape(self.inputs)[:1]

            m_lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(FLAGS.hidden_dim)
            m_c_init = np.zeros((1, FLAGS.hidden_dim * FLAGS.manager_horizon), np.float32)
            m_h_init = np.zeros((1, FLAGS.hidden_dim * FLAGS.manager_horizon), np.float32)
            self.m_state_init = [m_c_init, m_h_init]
            m_c_in = tf.placeholder(tf.float32, [1, FLAGS.hidden_dim * FLAGS.manager_horizon], name="Mrnn_c_in")
            m_h_in = tf.placeholder(tf.float32, [1, FLAGS.hidden_dim * FLAGS.manager_horizon], name="Mrnn_h_in")
            self.m_state_in = (m_c_in, m_h_in)
            m_state_in = tf.contrib.rnn.LSTMStateTuple(m_c_in, m_h_in)

            m_lstm_outputs, m_lstm_state = self.fast_dlstm(m_rnn_in, m_state_in, m_lstm_cell, FLAGS.manager_horizon,
                                                           FLAGS.hidden_dim * FLAGS.manager_horizon)

            m_lstm_c, m_lstm_h = m_lstm_state
            self.m_state_out = (m_lstm_c[-1, :1, :], m_lstm_h[-1, :1, :])
            m_rnn_out = tf.reshape(m_lstm_outputs, [-1, FLAGS.hidden_dim])

            self.goals = m_rnn_out

            self.normalized_goals = tf.nn.l2_normalize(self.goals, 1, name="Gt")

            summary_goals = tf.contrib.layers.summarize_activation(self.normalized_goals)

            def randomize_goals(t):
                t = tf.cast(t, tf.int32)
                packed_tensors = tf.stack([tf.random_normal([FLAGS.hidden_dim, ]), self.normalized_goals[t, :]])

                to_update = tf.cond(
                    tf.less(self.prob_of_random_goal, tf.constant(FLAGS.final_random_goal_prob, dtype=tf.float32)),
                    lambda: tf.cast(
                        tf.multinomial(
                            tf.log([[self.prob_of_random_goal,
                                     tf.subtract(tf.constant(1.0),
                                                 self.prob_of_random_goal)]]), 1)[0][0], tf.int32),
                    lambda: tf.constant(1, tf.int32))

                resulted_tensor = tf.gather(packed_tensors, to_update)

                return resulted_tensor

            self.randomized_goals = tf.map_fn(lambda t: randomize_goals(t), tf.to_float(tf.range(0, step_size[0])),
                                              name="random_gt")

            summary_random_goals = tf.contrib.layers.summarize_activation(self.randomized_goals)

            self.decrease_prob_of_random_goal = tf.assign_sub(self.prob_of_random_goal, tf.constant(
                (FLAGS.initial_random_goal_prob - FLAGS.final_random_goal_prob) / FLAGS.explore_steps))

            m_fc_value_w = tf.get_variable("M_Value_W", shape=[FLAGS.hidden_dim, 1],
                                           initializer=normalized_columns_initializer(1.0))
            self.m_value = tf.matmul(m_rnn_out, m_fc_value_w, name="M_Value")

            summary_m_value_act = tf.contrib.layers.summarize_activation(self.m_value)

            def gather_horiz(t):
                t = tf.cast(t, tf.int32)
                indices = tf.range(tf.maximum(t - tf.constant(FLAGS.manager_horizon), 0), t + 1)

                return tf.reduce_sum(tf.gather(tf.stop_gradient(self.randomized_goals), indices), axis=0)

            # with tf.control_dependencies([decrease_prob_of_random_goal]):
            # prev_goals_init = np.zeros((FLAGS.manager_horizon, FLAGS.hidden_dim), np.float32)
            # self.sum_prev_goals = tf.placeholder(tf.float32, [None, FLAGS.hidden_dim], name="sum_prev_goals")

            # self.sum_prev_goals = tf.map_fn(lambda t: gather_horiz(t), tf.to_float(tf.range(0, step_size[0])),
            #                                 name="sum_prev_goals")

            ############################################################################################################

            # Worker network
            self.sum_prev_goals = tf.placeholder(shape=[None, FLAGS.hidden_dim], dtype=tf.float32, name="Prev_c_Goals_sum")

            w_rnn_in = tf.expand_dims(self.f_percept, [0], name="Wrnn_in")
            step_size = tf.shape(self.inputs)[:1]
            w_lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(FLAGS.goal_embedding_size * FLAGS.nb_actions)
            w_c_init = np.zeros((1, w_lstm_cell.state_size.c), np.float32)
            w_h_init = np.zeros((1, w_lstm_cell.state_size.h), np.float32)
            self.w_state_init = [w_c_init, w_h_init]
            w_c_in = tf.placeholder(tf.float32, [1, w_lstm_cell.state_size.c], name="Wrnn_c_in")
            w_h_in = tf.placeholder(tf.float32, [1, w_lstm_cell.state_size.h], name="Wrnn_h_in")
            self.w_state_in = (w_c_in, w_h_in)
            w_state_in = tf.contrib.rnn.LSTMStateTuple(w_c_in, w_h_in)

            w_lstm_outputs, w_lstm_state = tf.nn.dynamic_rnn(
                w_lstm_cell, w_rnn_in, initial_state=w_state_in, sequence_length=step_size,
                time_major=False)

            w_lstm_c, w_lstm_h = w_lstm_state
            self.w_state_out = (w_lstm_c[:1, :], w_lstm_h[:1, :])
            Ut = tf.reshape(w_lstm_outputs, [step_size[0], FLAGS.nb_actions, FLAGS.goal_embedding_size],
                                   name="Ut")
            Ut_flat = tf.reshape(w_lstm_outputs, [step_size[0], FLAGS.nb_actions * FLAGS.goal_embedding_size],
                                        name="Ut_flat")

            summary_wrnn_act = tf.contrib.layers.summarize_activation(Ut)

            goal_encoding = tf.contrib.layers.fully_connected(self.sum_prev_goals, FLAGS.goal_embedding_size,
                                                              biases_initializer=None, scope="goal_emb")

            interm_rez = tf.squeeze(tf.matmul(Ut, tf.expand_dims(goal_encoding, 2)), 2)
            interm_rez = tf.contrib.layers.flatten(interm_rez)
            self.w_policy = tf.nn.softmax(interm_rez, name="W_Policy")

            summary_w_policy_act = tf.contrib.layers.summarize_activation(self.w_policy)

            w_fc_value_w = tf.get_variable("W_Value_W", shape=[FLAGS.nb_actions * FLAGS.goal_embedding_size + FLAGS.goal_embedding_size, 1],
                                           initializer=normalized_columns_initializer(1.0))
            self.w_value = tf.matmul(tf.concat([Ut_flat, goal_encoding], 1), w_fc_value_w, name="W_Value")
            # w_fc_value_w = tf.get_variable("W_FC_Value_W", shape=[
            #     FLAGS.nb_actions , 1],
            #                               initializer=normalized_columns_initializer(1.0))
            # self.w_value = tf.matmul(interm_rez, w_fc_value_w, name="W_Value")

            summary_w_value_act = tf.contrib.layers.summarize_activation(self.w_value)

            if scope != 'global':
                # def calculate_intrinsic_reward(t):
                #     indices = tf.range(tf.maximum(t - tf.constant(FLAGS.manager_horizon), 0), tf.maximum(t, 0))
                #     original_state = tf.map_fn(lambda i: self.f_Mspace[t, :], tf.to_float(indices))
                #     goals = tf.gather(self.randomized_goals, indices)
                #     state_diff = original_state - tf.gather(self.f_Mspace, indices)
                #     intrinsic_reward = tf.cast((1 / tf.shape(state_diff)[0]), tf.float32) * tf.reduce_sum(
                #         self.cosine_distance(state_diff, goals, dim=1))
                #
                #     return intrinsic_reward
                #
                # def gather_intrinsic_rewards(t):
                #     t = tf.cast(t, tf.int32)
                #
                #     intrinsic_reward = tf.cond(tf.cast(tf.equal(t, tf.constant(0)), tf.bool),
                #                                lambda: tf.constant(0.0),
                #                                lambda: calculate_intrinsic_reward(t))
                #
                #     return intrinsic_reward
                #
                # self.intr_rewards = tf.cast(
                #     tf.map_fn(lambda t: gather_intrinsic_rewards(t), tf.to_float(tf.range(0, step_size[0])),
                #               name="intrinsic_rewards"), dtype=tf.float32)

                # self.intr_rewards = tf.placeholder(shape=[None], dtype=tf.float32)

                self.w_extrinsic_return = tf.placeholder(shape=[None], dtype=tf.float32)
                self.m_extrinsic_return = tf.placeholder(shape=[None], dtype=tf.float32)
                self.w_intrinsic_return = tf.placeholder(shape=[None], dtype=tf.float32)

                def gather_state_at_horiz(t):
                    t = tf.cast(t, tf.int32)
                    f_Mspace_c = tf.gather(self.f_Mspace,
                                           tf.minimum(t + tf.constant(FLAGS.manager_horizon, dtype=tf.int32),
                                                      step_size[0] - 1))
                    return f_Mspace_c

                self.f_Mspace_c = tf.cast(
                    tf.map_fn(lambda t: gather_state_at_horiz(t), tf.to_float(tf.range(0, step_size[0])),
                              name="state_at_horiz"), dtype=tf.float32)
                self.state_diff = self.f_Mspace_c - self.f_Mspace
                self.cos_sim_state_diff = self.cosine_distance(tf.stop_gradient(self.state_diff), self.normalized_goals,
                                                               dim=1)

                self.m_advantages = self.m_extrinsic_return - tf.stop_gradient(tf.reshape(self.m_value, [-1]))
                self.goals_loss = tf.reduce_sum(self.m_advantages * self.cos_sim_state_diff)
                self.m_value_loss = FLAGS.m_beta_v * tf.reduce_sum(
                    tf.square(self.m_extrinsic_return - tf.reshape(self.m_value, [-1])))

                self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")
                self.actions_onehot = tf.one_hot(self.actions, FLAGS.nb_actions, dtype=tf.float32,
                                                 name="Actions_Onehot")

                self.responsible_outputs = tf.reduce_sum(self.w_policy * self.actions_onehot, [1])

                # self.intr_rewards = tf.concat([self.intr_rewards, tf.constant(0.0, shape=(1,))], axis=0)
                # self.discounted_intrinsic_rewards = tf.scan(
                #     lambda a, x: tf.constant(FLAGS.w_gamma, dtype=tf.float32) * a + x, tf.reverse(self.intr_rewards, [0]),
                #     name="discounted_intr_rewards")
                # self.discounted_intrinsic_rewards = tf.reverse(self.discounted_intrinsic_rewards, [0])[:-1]
                self.intrinsic_return = FLAGS.alpha * self.w_intrinsic_return
                self.total_return = self.w_extrinsic_return + self.intrinsic_return
                self.w_advantages = self.total_return - tf.stop_gradient(tf.reshape(self.w_value, [-1]))

                # Loss functions
                self.w_value_loss = FLAGS.w_beta_v * tf.reduce_sum(
                    tf.square(self.total_return - tf.reshape(self.w_value, [-1])))
                self.entropy = - tf.reduce_sum(self.w_policy * tf.log(self.w_policy + 1e-7))

                self.w_policy_loss = -tf.reduce_sum(
                    tf.log(self.responsible_outputs + 1e-7) * self.w_advantages) - self.entropy * FLAGS.beta_e

                self.loss = self.w_value_loss + self.w_policy_loss + self.m_value_loss + self.goals_loss

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, FLAGS.gradient_clip_value)

                self.worker_summaries = [summary_f_percept_act, summary_f_Mspace_act, summary_goals,
                                         summary_random_goals,
                                         summary_m_value_act,
                                         summary_wrnn_act, summary_w_policy_act, summary_w_value_act]
                for grad, weight in zip(grads, local_vars):
                    self.worker_summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
                    self.worker_summaries.append(tf.summary.histogram(weight.name, weight))

                self.merged_summary = tf.summary.merge(self.worker_summaries)

                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

    # def conditional_backprop(self, do_backprop, tensor, previous_tensor):
    #     # do_backprop = tf.Print(do_backprop, [do_backprop], "switch query")
    #     t = tf.cond(tf.cast(do_backprop, tf.bool),
    #                 lambda: tf.identity(tensor),
    #                 lambda: tf.zeros_like(tensor))
    #
    #     #
    #     # t = tf.cond(tf.cast(do_backprop, tf.bool),
    #     #             lambda: tf.Print(tensor, [0],
    #     #                              "backprop enabled for " + tensor.op.name),
    #     #             lambda: tf.zeros_like(tensor))
    #     y = t + tf.stop_gradient(tensor - t)
    #     return y
    # def conditional_backprop(self, do_backprop, tensor, previous_tensor):
    #     t = tf.cond(tf.cast(do_backprop, tf.bool),
    #                 lambda: tf.identity(tensor),
    #                 lambda: tf.identity(previous_tensor))
    #
    #     return t
    def conditional_sub_state(self, is_this_current_step, tensor, previous_tensor):
        t = tf.cond(tf.cast(is_this_current_step, tf.bool),
                    lambda: tensor,
                    lambda: previous_tensor)

        return t

    def fast_dlstm(self, s_t, state_in, lstm, chunks, h_size):
        # def dilate_one_time_step(one_h, previous_one_h, switcher, num_chunks):
        #     h_slices = []
        #
        #     chunk_step_size = h_size // num_chunks
        #     for switch_step, h_step in zip(range(num_chunks), range(0, h_size, chunk_step_size)):
        #         one_switch = switcher[switch_step]
        #         h_s = self.conditional_backprop(one_switch, one_h[h_step: h_step + chunk_step_size],
        #                                         previous_one_h[h_step: h_step + chunk_step_size])
        #         h_slices.append(h_s)
        #     dh = tf.stack(h_slices)
        #     dh = tf.reshape(dh, [-1, h_size])
        #     return dh

        # lstm = tf.tensorflow.contrib.rnn.LSTMCell(256, state_is_tuple=True)
        # chunks = 8

        def get_sub_state(state, state_step):
            c, h = state
            chunk_step_size = h_size // chunks
            h_step = state_step * chunk_step_size
            sub_state_h = h[:, h_step: h_step + chunk_step_size]
            sub_state_c = c[:, h_step: h_step + chunk_step_size]
            sub_state_h.set_shape([1, chunk_step_size])
            sub_state_c.set_shape([1, chunk_step_size])
            sub_state = tf.contrib.rnn.LSTMStateTuple(sub_state_c, sub_state_h)
            return sub_state

        def build_new_state(new_sub_state, previous_state, state_step):
            c_previous_state, h_previous_state = previous_state
            c_new_sub_state, h_new_sub_state = new_sub_state
            h_slices = []
            c_slices = []
            chunk_step_size = h_size // chunks
            one_hot_state_step = tf.one_hot(state_step, depth=chunks)

            for switch_step, h_step in zip(range(chunks), range(0, h_size, chunk_step_size)):
                is_this_current_step = one_hot_state_step[switch_step]
                h_s = self.conditional_sub_state(is_this_current_step, h_new_sub_state,
                                                 h_previous_state[:, h_step: h_step + chunk_step_size])
                h_s.set_shape([1, chunk_step_size])
                c_s = self.conditional_sub_state(is_this_current_step,
                                                 c_new_sub_state,
                                                 c_previous_state[:, h_step: h_step + chunk_step_size])
                c_s.set_shape([1, chunk_step_size])
                h_slices.append(h_s)
                c_slices.append(c_s)
            h_new_state = tf.concat(h_slices, axis=1)
            c_new_state = tf.concat(c_slices, axis=1)
            new_state = tf.contrib.rnn.LSTMStateTuple(c_new_state, h_new_state)
            return new_state

        def dlstm_scan_fn(previous_output, current_input):
            # out, state_out = lstm(current_input, previous_output[1])
            state_step = previous_output[2]

            sub_state = get_sub_state(previous_output[1], state_step)
            out, sub_state_out = lstm(current_input, sub_state)
            state_out = build_new_state(sub_state_out, previous_output[1], state_step)
            state_step += tf.constant(1)
            new_state_step = tf.mod(state_step, chunks)


            # state_out_dilated = dilate_one_time_step(tf.squeeze(state_out[0]), tf.squeeze(previous_output[1][0]),
            #                                          basis_i, chunks)
            # state_out = tf.contrib.rnn.LSTMStateTuple(state_out_dilated, state_out[1])
            # i += tf.constant(1)
            # new_i = tf.mod(i, chunks)
            return out, state_out, new_state_step

        chunk_step_size = h_size // chunks
        first_input = state_in.c[:, 0: chunk_step_size]
        rnn_outputs, final_states, mod_idxs = tf.scan(dlstm_scan_fn,
                                                      tf.transpose(s_t, [1, 0, 2]),
                                                      initializer=(
                                                          first_input, state_in, tf.constant(0)), name="dlstm")

        # state_out = [final_states[0][-1, :1, :], final_states[1][-1, :1, :]]
        # cell_states = final_states[0][:, 0, :]
        # out_states = final_states[1][:, 0, :]
        return rnn_outputs, final_states

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

    def cosine_distance(self, v1, v2, dim):
        v1_norm = tf.nn.l2_normalize(v1, dim)
        v2_norm = tf.nn.l2_normalize(v2, dim)
        sim = tf.matmul(
            v1_norm, v2_norm, transpose_b=True)

        return sim
