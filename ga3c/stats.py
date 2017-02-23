from multiprocessing import Process, Queue, Value
import numpy as np
import flags
import gym
import gym_fast_envs
import tensorflow as tf
from atari_environment import AtariEnvironment
import time
from scipy.signal import lfilter
FLAGS = tf.app.flags.FLAGS

class Stats(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(Stats, self).__init__()

        self.episode_log_q = Queue(maxsize=100)
        self.episode_rewards = []
        self.episode_lengths = []

        self.episode_count = Value('i', 0)

        self.summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir)
        self.summary = tf.Summary()

    def run(self):
        while True:
            time_of_reward, episode_reward, episode_length = self.episode_log_q.get()
            self.episode_rewards.append([time_of_reward, episode_reward, episode_length])

            self.episode_count.value += 1

            if self.episode_count.value % FLAGS.summary_interval:
                mean_reward = np.mean(self.episode_rewards[-FLAGS.summary_interval:])
                mean_length = np.mean(self.episode_lengths[-FLAGS.summary_interval:])

                self.summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                self.summary.value.add(tag='Perf/Length', simple_value=float(mean_length))


