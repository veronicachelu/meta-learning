import threading
import multiprocessing
from threading import Lock
import gym
from gym import wrappers
import gym_maze
import gym_fast_envs
# import gym_ple
import tensorflow as tf
from agent import Agent
from atari_environment import AtariEnvironment
from network import DQNetwork
from tensorflow.python import debug as tf_debug
import flags

FLAGS = tf.app.flags.FLAGS

main_lock = Lock()


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

    sess = tf.Session()
    with sess:
        global_step = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)

        gym_env = gym.make(FLAGS.game)
        if FLAGS.seed and FLAGS.seed != -1:
            gym_env.seed(FLAGS.seed)

        if FLAGS.monitor:
           gym_env = gym.wrappers.Monitor(gym_env, FLAGS.experiments_dir)

        env = AtariEnvironment(gym_env=gym_env, resized_width=FLAGS.resized_width,
                                    resized_height=FLAGS.resized_height,
                                    agent_history_length=FLAGS.agent_history_length)
        nb_actions = len(env.gym_actions)

        agent = Agent(env, sess, nb_actions, global_step)
        saver = tf.train.Saver(max_to_keep=1000)

    if FLAGS.resume:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        print("Loading Model from {}".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    agent.play(saver)


if __name__ == '__main__':
    run()
