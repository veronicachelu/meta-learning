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
            # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
            optimizer = tf.train.RMSPropOptimizer(FLAGS.lr, 0.99, 0.0, 1e-6)

            # num_workers = FLAGS.nb_concurrent
            num_workers = multiprocessing.cpu_count() - 1
            workers = []
            envs = []

            for i in range(num_workers):
                gym_env = gym.make(FLAGS.game)
                gym_env.seed(FLAGS.seed)

                if FLAGS.monitor:
                    gym_env = gym.wrappers.Monitor(gym_env, FLAGS.experiments_dir + '/worker_{}'.format(i))
                this_env = AtariEnvironment(gym_env=gym_env, resized_width=FLAGS.resized_width,
                                            resized_height=FLAGS.resized_height,
                                            agent_history_length=FLAGS.agent_history_length)

                envs.append(this_env)
            nb_actions = len(envs[0].gym_actions)

            if FLAGS.lstm:
                global_network = ACNetworkLSTM('global', nb_actions, None)
            else:
                global_network = ACNetwork('global', nb_actions, None)

            for i in range(num_workers):
                workers.append(Worker(envs[i], sess, i, nb_actions, optimizer, global_step))
            saver = tf.train.Saver(max_to_keep=5)

            gym_env_monitor = gym.make(FLAGS.game)
            gym_env_monitor.seed(FLAGS.seed)
            gym_env_monitor_wrapper = AtariEnvironment(gym_env=gym_env_monitor, resized_width=FLAGS.resized_width,
                                        resized_height=FLAGS.resized_height,
                                        agent_history_length=FLAGS.agent_history_length)
            nb_actions = len(gym_env_monitor_wrapper.gym_actions)
            pe = PolicyMonitor(
                game=gym_env_monitor_wrapper,
                nb_actions=nb_actions,
                optimizer=optimizer,
                global_step=global_step
            )

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

        # Start a thread for policy eval task
        monitor_thread = threading.Thread(target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord))
        monitor_thread.start()
        import time
        while True:
            if FLAGS.show_training:
                for env in envs:
                    # time.sleep(1)
                    # with main_lock:
                    env.env.render()

        coord.join(worker_threads)


if __name__ == '__main__':
    run()
