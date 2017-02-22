import threading
import multiprocessing
from threading import Lock
import gym
from gym import wrappers
import gym_fast_envs
#import gym_ple
import tensorflow as tf
from agent import Worker
from atari_environment import AtariEnvironment
from network import ACNetwork
from network_lstm import ACNetworkLSTM
from eval import PolicyMonitor
import flags

FLAGS = tf.app.flags.FLAGS

main_lock = Lock()

def run():

    tf.reset_default_graph()

    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            global_step = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
            # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
            optimizer = tf.train.RMSPropOptimizer(FLAGS.lr, 0.99, 0.0, 1e-6)
            gym_env_monitor = gym.make(FLAGS.game)
            gym_env_monitor.seed(FLAGS.seed)
            gym_env_monitor_wrapper = AtariEnvironment(gym_env=gym_env_monitor, resized_width=FLAGS.resized_width,
                                                       resized_height=FLAGS.resized_height,
                                                       agent_history_length=FLAGS.agent_history_length)
            nb_actions = len(gym_env_monitor_wrapper.gym_actions)

            if FLAGS.lstm:
                global_network = ACNetworkLSTM('global', nb_actions, None)
            else:
                global_network = ACNetwork('global', nb_actions, None)


            saver = tf.train.Saver(max_to_keep=5)

        if FLAGS.resume:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            print("Loading Model from {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        pe = PolicyMonitor(
            game=gym_env_monitor_wrapper,
            nb_actions=nb_actions,
            optimizer=optimizer,
            global_step=global_step
        )
        pe.eval_1000(sess)


if __name__ == '__main__':
    run()
