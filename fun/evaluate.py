import threading

import tensorflow as tf
import random
import numpy as np
import gym
from gym import wrappers
import gym_fast_envs
from agent import Agent
from network import FUNNetwork
import flags
import multiprocessing
import os
FLAGS = tf.app.flags.FLAGS
from eval import PolicyMonitor



# def sample_params():
#     FLAGS.lr = 10 ** np.random.uniform(np.log10(10**(-2)), np.log10((10**(-4))))
#     FLAGS.gamma = np.random.uniform(0.8, 1.0)

def run():
    if not tf.gfile.Exists(os.path.join(FLAGS.frames_dir, FLAGS.model_name)):
        tf.gfile.DeleteRecursively(os.path.join(FLAGS.frames_dir, FLAGS.model_name))
        tf.gfile.MakeDirs(os.path.join(FLAGS.frames_dir, FLAGS.model_name))


    tf.reset_default_graph()

    sess = tf.Session()
    with sess:
        with tf.device("/cpu:0"):
            global_step = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
            global_network = FUNNetwork('global', None)

            saver = tf.train.Saver(max_to_keep=5)
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name))
            print("Loading Model from {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            sess.run(tf.local_variables_initializer())
            gym_env_monitor = gym.make(FLAGS.game)
            if FLAGS.monitor:
                gym_env_monitor = gym.wrappers.Monitor(gym_env_monitor,
                                                       os.path.join(FLAGS.test_experiments_dir, FLAGS.model_name),
                                                       force=True)

            pe = PolicyMonitor(
                game=gym_env_monitor,
                optimizer=optimizer,
                global_step=global_step
            )

        coord = tf.train.Coordinator()
        # Start a thread for policy eval task
        monitor_thread = threading.Thread(target=lambda: pe.eval_nb_test_episodes(sess))
        monitor_thread.start()
        import time

        coord.join([monitor_thread])


if __name__ == '__main__':
    run()
