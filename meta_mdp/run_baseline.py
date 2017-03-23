import threading

import tensorflow as tf
import random
import numpy as np
import gym
from gym import wrappers
import gym_fast_envs
from agent import Agent
from random_agent import RandomAgent
from network import ACNetwork
import flags
import multiprocessing
FLAGS = tf.app.flags.FLAGS


def run():
    tf.reset_default_graph()

    sess = tf.Session()
    with sess:
        with tf.device("/cpu:0"):
            gym_env = gym.make(FLAGS.game)
            if FLAGS.monitor:
                gym_env = gym.wrappers.Monitor(gym_env, FLAGS.experiments_dir + '/baseline', force=True)

            agent = RandomAgent(gym_env)

            agent.play()

if __name__ == '__main__':
    run()
