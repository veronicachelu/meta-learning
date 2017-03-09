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
import os
FLAGS = tf.app.flags.FLAGS


def recreate_directory_structure():
    if not tf.gfile.Exists(os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name)):
        tf.gfile.MakeDirs(os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name))
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name))
            tf.gfile.MakeDirs(os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name))

    if not tf.gfile.Exists(os.path.join(FLAGS.experiments_dir, FLAGS.model_name)):
        tf.gfile.MakeDirs(os.path.join(FLAGS.experiments_dir, FLAGS.model_name))
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(os.path.join(FLAGS.experiments_dir, FLAGS.model_name))
            tf.gfile.MakeDirs(os.path.join(FLAGS.experiments_dir, FLAGS.model_name))

    if not tf.gfile.Exists(os.path.join(FLAGS.summaries_dir, FLAGS.model_name)):
        tf.gfile.MakeDirs(os.path.join(FLAGS.summaries_dir, FLAGS.model_name))
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(os.path.join(FLAGS.summaries_dir, FLAGS.model_name))
            tf.gfile.MakeDirs(os.path.join(FLAGS.summaries_dir, FLAGS.model_name))

# def sample_params():
#     FLAGS.lr = 10 ** np.random.uniform(np.log10(10**(-2)), np.log10((10**(-4))))
#     FLAGS.gamma = np.random.uniform(0.8, 1.0)

def run():
    recreate_directory_structure()
    tf.reset_default_graph()

    sess = tf.Session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    with sess:
        with tf.device("/cpu:0"):
            global_step = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
            global_network = ACNetwork('global', None)

            # num_agents = multiprocessing.cpu_count()
            num_agents = FLAGS.nb_concurrent
            agents = []
            envs = []

            for i in range(num_agents):
                gym_env = gym.make(FLAGS.game)
                # if FLAGS.monitor:
                #     gym_env = gym.wrappers.Monitor(gym_env, FLAGS.experiments_dir + '/worker_{}'.format(i), force=True)
                envs.append(gym_env)

            for i in range(num_agents):
                agents.append(Agent(envs[i], i, optimizer, global_step))
            saver = tf.train.Saver(max_to_keep=5)

        coord = tf.train.Coordinator()
        if FLAGS.resume:
            ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name))
            print("Loading Model from {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        agent_threads = []
        for agent in agents:
            thread = threading.Thread(target=(lambda: agent.play(sess, coord, saver)))
            thread.start()
            agent_threads.append(thread)

        while True:
            if FLAGS.show_training:
                for env in envs:
                    # time.sleep(1)
                    # with main_lock:
                    env.render()

        coord.join(agent_threads)


if __name__ == '__main__':
    run()
