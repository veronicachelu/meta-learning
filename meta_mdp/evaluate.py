import threading

import tensorflow as tf
import random
import numpy as np
import gym
from gym import wrappers
import gym_fast_envs
from agent import Agent
from network import ACNetwork
import flags
import multiprocessing
from eval import PolicyMonitor
FLAGS = tf.app.flags.FLAGS

# def sample_params():
#     FLAGS.lr = 10 ** np.random.uniform(np.log10(10**(-2)), np.log10((10**(-4))))
#     FLAGS.gamma = np.random.uniform(0.8, 1.0)

def run():
    tf.reset_default_graph()

    sess = tf.Session()
    with sess:
        with tf.device("/cpu:0"):
            global_step = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
            global_network = ACNetwork('global', None)

            gym_env_monitor = gym.make(FLAGS.game)
            pe = PolicyMonitor(
                game=gym_env_monitor,
                optimizer=optimizer,
                global_step=global_step
            )
            saver = tf.train.Saver(max_to_keep=5)

        coord = tf.train.Coordinator()
        if FLAGS.resume:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            print("Loading Model from {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        #Start a thread for policy eval task
        monitor_thread = threading.Thread(target=lambda: pe.eval_nb_test_episodes(sess))
        monitor_thread.start()

        coord.join([monitor_thread])


if __name__ == '__main__':
    run()
