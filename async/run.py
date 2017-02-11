import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from random import choice
from time import sleep
from time import time
from atari_environment import AtariEnvironment
from network import AC_Network
from agent import Worker
import flags
import gym

FLAGS = tf.app.flags.FLAGS
# gamma = .99 # discount rate for advantage estimation and reward discounting
# s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
# a_size = 3 # Agent can move Left, Right, or Fire
# load_model = True
# model_path = './model'

tf.reset_default_graph()

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

# Create a directory to save episode playback gifs to
if not os.path.exists(FLAGS.frames_dir):
    os.makedirs(FLAGS.frames_dir)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
    master_network = AC_Network('global', None)  # Generate global network
    num_workers = FLAGS.nb_concurrent  # multiprocessing.cpu_count()  # Set workers ot number of available CPU threads
    workers = []
    envs = []
    # Create worker classes
    for i in range(num_workers):
        envs.append(gym.make(FLAGS.game))
    for i in range(num_workers):
        workers.append(Worker(envs[i], i, trainer, FLAGS.checkpoint_dir, global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if FLAGS.resume == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        worker_threads.append(t)

    # while True:
    #     if FLAGS.show_training:
    #         for env in envs:
    #             env.render()

    coord.join(worker_threads)
