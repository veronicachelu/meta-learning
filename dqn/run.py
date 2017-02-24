import threading
import multiprocessing
from threading import Lock
import gym
from gym import wrappers
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
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    with sess:
        global_step = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)

        gym_env = gym.make(FLAGS.game)
        gym_env.seed(FLAGS.seed)

        if FLAGS.monitor:
            gym_env = gym.wrappers.Monitor(gym_env, FLAGS.experiments_dir + '/worker_{}'.format(i))
        env = AtariEnvironment(gym_env=gym_env, resized_width=FLAGS.resized_width,
                                    resized_height=FLAGS.resized_height,
                                    agent_history_length=FLAGS.agent_history_length)

        nb_actions = len(env.gym_actions)

        agent = Agent(env, sess, nb_actions, global_step)
        saver = tf.train.Saver(max_to_keep=5)

        # gym_env_monitor = gym.make(FLAGS.game)
        # gym_env_monitor.seed(FLAGS.seed)
        # gym_env_monitor_wrapper = AtariEnvironment(gym_env=gym_env_monitor, resized_width=FLAGS.resized_width,
        #                                            resized_height=FLAGS.resized_height,
        #                                            agent_history_length=FLAGS.agent_history_length)
        # nb_actions = len(gym_env_monitor_wrapper.gym_actions)
        # pe = PolicyMonitor(
        #     game=gym_env_monitor_wrapper,
        #     nb_actions=nb_actions,
        #     optimizer=optimizer,
        #     global_step=global_step
        # )

    # coord = tf.train.Coordinator()
    if FLAGS.resume:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        print("Loading Model from {}".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    agent.play(saver)

    # # Start a thread for policy eval task
    # monitor_thread = threading.Thread(target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord))
    # monitor_thread.start()
    # import time
    # while True:
    #     if FLAGS.show_training:
    #         for env in envs:
    #             # time.sleep(1)
    #             # with main_lock:
    #             env.env.render()
    #
    # coord.join(worker_threads)


if __name__ == '__main__':
    run()
