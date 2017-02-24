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
    def __init__(self):
        super(Stats, self).__init__(name="Stats")

        self.episode_log_q = Queue(maxsize=100)
        self.episode_rewards = []
        self.episode_lengths = []

        self.episode_count = Value('i', 0)

        self.summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir)
        self.summary = tf.Summary()

    def run(self):
        while True:
            print("Stats thread takes a tuple from the log queue. Episode count it {}".format(self.episode_count.value))
            time_of_reward, episode_reward, episode_length = self.episode_log_q.get()
            self.episode_rewards.append([time_of_reward, episode_reward, episode_length])

            self.episode_count.value += 1

            if self.episode_count.value % FLAGS.summary_interval:
                print("Stats thread makes a new summary log")
                mean_reward = np.mean(self.episode_rewards[-FLAGS.summary_interval:])
                mean_length = np.mean(self.episode_lengths[-FLAGS.summary_interval:])

                self.summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                self.summary.value.add(tag='Perf/Length', simple_value=float(mean_length))

                self.summary_writer.add_summary(self.summary, self.episode_count.value)
                self.summary_writer.flush()

            time.sleep(0.05)
