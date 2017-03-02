import threading

import tensorflow as tf
import random
import numpy as np
from agent import Worker
from envs.bandit_envs import TwoArms, ElevenArms
from network import AC_Network
import flags

FLAGS = tf.app.flags.FLAGS


def recreate_directory_structure():
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
            tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    if not tf.gfile.Exists(FLAGS.frames_dir):
        tf.gfile.MakeDirs(FLAGS.frames_dir)
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(FLAGS.frames_dir)
            tf.gfile.MakeDirs(FLAGS.frames_dir)

    if not tf.gfile.Exists(FLAGS.frames_test_dir):
        tf.gfile.MakeDirs(FLAGS.frames_test_dir)
    else:
        if FLAGS.resume and not FLAGS.train:
            tf.gfile.DeleteRecursively(FLAGS.frames_test_dir)
            tf.gfile.MakeDirs(FLAGS.frames_test_dir)

    if not tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.MakeDirs(FLAGS.summaries_dir)
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
            tf.gfile.MakeDirs(FLAGS.summaries_dir)

# def sample_params():
#     FLAGS.lr = 10 ** np.random.uniform(np.log10(10**(-2)), np.log10((10**(-4))))
#     FLAGS.gamma = np.random.uniform(0.8, 1.0)

def run():
    recreate_directory_structure()
    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        global_step = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
        global_network = AC_Network('global', None)

        num_workers = 1
        workers = []
        envs = []
        for i in range(num_workers):
            if FLAGS.game == '11arms':
                this_env = ElevenArms()
            else:
                this_env = TwoArms(FLAGS.game)
            envs.append(this_env)

        for i in range(num_workers):
            workers.append(Worker(envs[i], i, optimizer, global_step))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if FLAGS.resume:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            print("Loading Model from {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
            worker_play = lambda: worker.play(sess, coord, saver)
            thread = threading.Thread(target=(worker_play))
            thread.start()
            worker_threads.append(thread)
        coord.join(worker_threads)


if __name__ == '__main__':
    run()
