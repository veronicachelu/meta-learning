import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf
import time

from gym.wrappers import Monitor
from atari_environment import AtariEnvironment
from network import ACNetwork
from utils import update_target_graph
import gym
import flags

FLAGS = tf.app.flags.FLAGS

class PolicyMonitor(object):
    def __init__(self, game, nb_actions, optimizer, global_step):
        self.name = "policy_eval"
        self.local_AC = ACNetwork(self.name, nb_actions, optimizer)
        self.update_local_ops = update_target_graph('global', self.name)
        self.summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/policy_eval")
        self.env = game
        self.actions = np.zeros([nb_actions])
        self.global_episode = global_step

    def eval_once(self, sess):
        with sess.as_default(), sess.graph.as_default():
            # Copy params to local model
            episode_count = sess.run(self.global_episode)
            sess.run(self.update_local_ops)

            # Run an episode
            d = False
            s = self.env.get_initial_state()

            total_reward = 0.0
            episode_length = 0
            while not d:
                feed_dict = {self.local_AC.inputs: [s]}
                pi, v = sess.run(
                    [self.local_AC.policy, self.local_AC.value],
                    feed_dict=feed_dict)

                a = np.random.choice(pi[0], p=pi[0])
                a = np.argmax(pi == a)

                s1, r, d, info = self.env.step(a)

                total_reward += r
                episode_length += 1
                s = s1

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
