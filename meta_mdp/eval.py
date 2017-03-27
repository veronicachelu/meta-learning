import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf
import time

from network import ACNetwork, ConvNetwork
from utils import update_target_graph
import gym
import flags

FLAGS = tf.app.flags.FLAGS


class PolicyMonitor(object):
    def __init__(self, game, optimizer, global_step):
        self.name = "policy_eval"
        if FLAGS.use_conv:
            self.local_AC = ConvNetwork(self.name, optimizer, global_step)
        else:
            self.local_AC = ACNetwork(self.name, optimizer, global_step)
        self.update_local_ops = update_target_graph('global', self.name)
        self.summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/policy_eval")
        self.env = game
        # self.actions = np.zeros([nb_actions])
        self.global_episode = global_step

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

            # Run an episode
            d = False
            s, _, _, _ = self.env.reset()
            r = 0
            a = 0
            t = 0
            total_reward = 0.0
            episode_length = 0

            rnn_state = self.local_AC.state_init

            while not d:
                if FLAGS.meta:
                    feed_dict = {
                        self.local_AC.inputs: [s],
                        self.local_AC.prev_rewards: [[r]],
                        self.local_AC.timestep: [[t]],
                        self.local_AC.prev_actions: [a],
                        self.local_AC.state_in[0]: rnn_state[0],
                        self.local_AC.state_in[1]: rnn_state[1]}
                else:
                    feed_dict = {
                        self.local_AC.inputs: [s],
                        self.local_AC.state_in[0]: rnn_state[0],
                        self.local_AC.state_in[1]: rnn_state[1]}

                pi, v, rnn_state_new = sess.run(
                    [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out], feed_dict=feed_dict)
                a = np.random.choice(pi[0], p=pi[0])
                a = np.argmax(pi == a)

                rnn_state = rnn_state_new
                s1, r, d, _ = self.env.step(a)

                total_reward += r
                episode_length += 1
                t += 1
                s = s1

                if t > 100:
                    d = True

            if summaries:
                # Add summaries
                episode_summary = tf.Summary()
                episode_summary.value.add(simple_value=total_reward, tag="eval/total_reward")
                episode_summary.value.add(simple_value=episode_length, tag="eval/episode_length")
                self.summary_writer.add_summary(episode_summary, episode_count)
                self.summary_writer.flush()

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
