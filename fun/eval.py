import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf
import time

from network import FUNNetwork
from utils import update_target_graph
import gym
import flags

FLAGS = tf.app.flags.FLAGS


class PolicyMonitor(object):
    def __init__(self, game, optimizer, global_step):
        self.name = "policy_eval"
        self.global_episode = global_step
        self.local_AC = FUNNetwork(self.name, optimizer, self.global_episode)

        self.update_local_ops = update_target_graph('global', self.name)
        self.summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/policy_eval")
        self.env = game
        # self.actions = np.zeros([nb_actions])


    def eval_nb_test_episodes(self, sess):
        rewards = []
        for i in range(FLAGS.nb_test_episodes):
            episode_reward, _ = self.eval_once(sess, False)
            rewards.append(episode_reward)
        print("Mean reward over {} episodes : {}".format(FLAGS.nb_test_episodes, np.mean(rewards)))

    def eval_once(self, sess, summaries=True):
        with sess.as_default(), sess.graph.as_default():
            # Copy params to local model
            episode_count = sess.run(self.global_episode)
            sess.run(self.update_local_ops)
            sess.run(self.local_AC.decrease_prob_of_random_goal)

            # Run an episode
            s, _, _, _ = self.env.reset()
            d = False
            t = 0
            r = 0
            a = 0
            prev_goal = np.random.normal(size=(FLAGS.hidden_dim,))
            m_rnn_state = self.local_AC.m_state_init
            w_rnn_state = self.local_AC.w_state_init

            total_reward = 0.0
            episode_length = 0

            episode_buffer = []

            episode_w_values = []
            episode_intr_reward = []
            episode_m_values = []
            episode_reward = 0
            episode_step_count = 0
            episode_goals = []
            episode_sum_of_prev_goals = []
            episode_manager_states = []


            while not d:
                feed_dict_m = {
                    self.local_AC.inputs: [s],
                    self.local_AC.prev_rewards: [r],
                    self.local_AC.prev_goal: [prev_goal],
                    self.local_AC.m_state_in[0]: m_rnn_state[0],
                    self.local_AC.m_state_in[1]: m_rnn_state[1]
                }

                m_v, m_rnn_state_new, goals, m_s = sess.run(
                    [self.local_AC.m_value, self.local_AC.m_state_out, self.local_AC.randomized_goals,
                     self.local_AC.f_Mspace], feed_dict=feed_dict_m)

                episode_goals.append(goals[0])
                episode_manager_states.append(m_s[0])

                def prev_goals_gather_horiz():
                    t = len(episode_goals)
                    s = 0
                    for i in range(max(t - FLAGS.manager_horizon, 0), t):
                        s += episode_goals[i]

                    return s

                def intr_reward_gather_horiz():
                    t = len(episode_manager_states)
                    s = 0
                    if t - 1 > 0:
                        for i in range(max(t - FLAGS.manager_horizon, 0), t - 1):
                            state_dif = episode_manager_states[t - 1] - episode_manager_states[i]
                            state_dif_norm = np.linalg.norm(state_dif)
                            if state_dif_norm != 0:
                                state_dif_normalized = state_dif / state_dif_norm
                            else:
                                state_dif_normalized = state_dif
                            goal_norm = np.linalg.norm(episode_goals[i])
                            if goal_norm != 0:
                                goal_normalized = episode_goals[i] / goal_norm
                            else:
                                goal_normalized = episode_goals[i]
                            s += np.dot(state_dif_normalized, goal_normalized)
                        s /= len(range(max(t - FLAGS.manager_horizon, 0), t - 1))
                    return s

                intr_reward = intr_reward_gather_horiz()
                episode_intr_reward.append(intr_reward)

                sum_of_prev_goals = prev_goals_gather_horiz()
                prev_goal = sum_of_prev_goals
                episode_sum_of_prev_goals.append(sum_of_prev_goals)

                feed_dict_w = {
                    self.local_AC.inputs: [s],
                    self.local_AC.prev_rewards: [r],
                    self.local_AC.prev_actions: [a],
                    self.local_AC.sum_prev_goals: [sum_of_prev_goals],
                    self.local_AC.w_state_in[0]: w_rnn_state[0],
                    self.local_AC.w_state_in[1]: w_rnn_state[1],
                    self.local_AC.m_state_in[0]: m_rnn_state[0],
                    self.local_AC.m_state_in[1]: m_rnn_state[1]
                }

                pi, w_v, w_rnn_state_new = sess.run(
                    [self.local_AC.w_policy, self.local_AC.w_value, self.local_AC.w_state_out], feed_dict=feed_dict_w)
                a = np.random.choice(pi[0], p=pi[0])
                a = np.argmax(pi == a)

                w_rnn_state = w_rnn_state_new
                m_rnn_state = m_rnn_state_new

                s1, r, d, _ = self.env.step(a)

                total_reward += r
                episode_length += 1
                t += 1
                s = s1

                if t >= FLAGS.BTT_length:
                    d = True

            if summaries:
                # # Add summaries
                # episode_summary = tf.Summary()
                # episode_summary.value.add(simple_value=total_reward, tag="eval/total_reward")
                # episode_summary.value.add(simple_value=episode_length, tag="eval/episode_length")
                # self.summary_writer.add_summary(episode_summary, episode_count)
                # self.summary_writer.flush()

                tf.logging.info(
                    "Eval results at step {}: total_reward {}, episode_length {}".format(episode_count, total_reward,
                                                                                         episode_length))

            return total_reward, episode_length

    def continuous_eval(self, eval_every, sess, coord):
        """
        Continuously evaluates the policy every [eval_every] seconds.
        """
        try:
            while not coord.should_stop():
                self.eval_once(sess)
                # Sleep until next evaluation cycle
                time.sleep(eval_every)
        except tf.errors.CancelledError:
            return
