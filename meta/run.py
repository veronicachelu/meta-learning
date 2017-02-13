import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np

from time import sleep
from time import time
from network import AC_Network
from agent import Worker
import flags
from envs.dependent_bandits import dependent_bandit, eleven_arms


FLAGS = tf.app.flags.FLAGS

tf.reset_default_graph()

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

# Create a directory to save episode playback gifs to
if not os.path.exists(FLAGS.frames_dir):
    os.makedirs(FLAGS.frames_dir)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
    master_network = AC_Network('global', None)  # Generate global network
    # num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
    num_workers = 1
    workers = []
    envs = []
    for i in range(num_workers):
        if FLAGS.game == '11arms':
            this_env = eleven_arms()
        else:
            this_env = dependent_bandit(FLAGS.game)
        envs.append(this_env)
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(envs[i], i, trainer, global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if FLAGS.resume == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        print("ckpt.model_checkpoint_path: {}".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(sess, coord, saver)
        thread = threading.Thread(target=(worker_work))
        thread.start()
        worker_threads.append(thread)
    coord.join(worker_threads)