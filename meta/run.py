import os
import threading

import tensorflow as tf
from agent import Worker
from envs.dependent_bandits import dependent_bandit, eleven_arms
from network import AC_Network

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

    # Create a directory to save episode playback gifs to
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

def run():
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
                this_env = eleven_arms()
            else:
                this_env = dependent_bandit(FLAGS.game)
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