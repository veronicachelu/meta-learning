import threading
import multiprocessing
from threading import Lock
import gym
from gym import wrappers
import gym_fast_envs
# import gym_ple
import tensorflow as tf
from agent import Worker
from atari_environment import AtariEnvironment
from network import ACNetwork
from network_lstm import ACNetworkLSTM
from eval import PolicyMonitor
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


if __name__ == '__main__':
    run()
