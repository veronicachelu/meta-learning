import threading

import tensorflow as tf
import random
import numpy as np
from agent import Agent
from envs.bandit_envs import TwoArms, ElevenArms
from network import ACNetwork
from baseline import RandomAgent
import flags
import os

FLAGS = tf.app.flags.FLAGS


def recreate_directory_structure():
    if not tf.gfile.Exists(os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name)):
        tf.gfile.MakeDirs(os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name))
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name))
            tf.gfile.MakeDirs(os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name))

    if not tf.gfile.Exists(os.path.join(FLAGS.frames_dir, FLAGS.model_name)):
        tf.gfile.MakeDirs(os.path.join(FLAGS.frames_dir, FLAGS.model_name))
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(os.path.join(FLAGS.frames_dir, FLAGS.model_name))
            tf.gfile.MakeDirs(os.path.join(FLAGS.frames_dir, FLAGS.model_name))

    if not tf.gfile.Exists(os.path.join(FLAGS.summaries_dir, FLAGS.model_name)):
        tf.gfile.MakeDirs(os.path.join(FLAGS.summaries_dir, FLAGS.model_name))
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(os.path.join(FLAGS.summaries_dir, FLAGS.model_name))
            tf.gfile.MakeDirs(os.path.join(FLAGS.summaries_dir, FLAGS.model_name))


def run_once():
    recreate_directory_structure()

    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name)
    summaries_dir = os.path.join(FLAGS.summaries_dir, FLAGS.model_name)
    frames_dir = os.path.join(FLAGS.frames_dir, FLAGS.model_name)

    settings = {"lr": FLAGS.lr,
                "gamma": FLAGS.gamma,
                "game": FLAGS.game,
                "model_name": FLAGS.model_name,
                "checkpoint_dir": checkpoint_dir,
                "summaries_dir": summaries_dir,
                "frames_dir": frames_dir}

    run(settings)


def run(settings):
    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        global_step = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=settings["lr"])
        global_network = ACNetwork('global', None)

        num_agents = 1
        agents = []
        envs = []
        for i in range(num_agents):
            if settings["game"] == '11arms':
                this_env = ElevenArms()
            else:
                this_env = TwoArms(settings["game"])
            envs.append(this_env)

        for i in range(num_agents):
            agents.append(Agent(envs[i], i, optimizer, global_step, settings))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if FLAGS.resume:
            ckpt = tf.train.get_checkpoint_state(settings["load_from"])
            print("Loading Model from {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        agent_threads = []
        for agent in agents:
            agent_play = lambda: agent.play(sess, coord, saver)
            thread = threading.Thread(target=agent_play)
            thread.start()
            agent_threads.append(thread)
        coord.join(agent_threads)


if __name__ == '__main__':
    run_once()
