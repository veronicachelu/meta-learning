import threading

import gym
from gym import wrappers
#import gym_fast_envs
import gym_ple
import tensorflow as tf
from agent import Worker
from atari_environment import AtariEnvironment
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

    if not tf.gfile.Exists(FLAGS.experiments_dir):
        tf.gfile.MakeDirs(FLAGS.experiments_dir)
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(FLAGS.experiments_dir)
            tf.gfile.MakeDirs(FLAGS.experiments_dir)

    if not tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.MakeDirs(FLAGS.summaries_dir)
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
            tf.gfile.MakeDirs(FLAGS.summaries_dir)


def run():
    recreate_directory_structure()
    tf.reset_default_graph()

    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            global_step = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)

            num_workers = FLAGS.nb_concurrent
            workers = []
            envs = []


            for i in range(num_workers):
                gym_env = gym.make(FLAGS.game)
                if FLAGS.monitor:
                    gym_env = gym.wrappers.Monitor(gym_env, FLAGS.experiments_dir + '/worker_{}'.format(i))
                this_env = AtariEnvironment(gym_env=gym_env, resized_width=FLAGS.resized_width,
                                            resized_height=FLAGS.resized_height,
                                            agent_history_length=FLAGS.agent_history_length)

                envs.append(this_env)
            nb_actions = len(envs[0].gym_actions)

            global_network = AC_Network('global', nb_actions, None)

            for i in range(num_workers):
                workers.append(Worker(envs[i], sess, i, nb_actions, optimizer, global_step))
            saver = tf.train.Saver(max_to_keep=5)


        coord = tf.train.Coordinator()
        if FLAGS.resume:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            print("Loading Model from {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
            t = threading.Thread(target=(lambda: worker.play(coord, saver)))
            t.start()
            worker_threads.append(t)

        # while True:
        #     if FLAGS.show_training:
        #         for env in envs:
        #             env.render()

        coord.join(worker_threads)


if __name__ == '__main__':
    run()
