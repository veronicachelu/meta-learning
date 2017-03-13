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

def recreate_subdirectory_structure(settings):
    if not tf.gfile.Exists(settings["checkpoint_dir"]):
        tf.gfile.MakeDirs(settings["checkpoint_dir"])
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(settings["checkpoint_dir"])
            tf.gfile.MakeDirs(settings["checkpoint_dir"])

    if not tf.gfile.Exists(settings["frames_dir"]):
        tf.gfile.MakeDirs(settings["frames_dir"])
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(settings["frames_dir"])
            tf.gfile.MakeDirs(settings["frames_dir"])

    if not tf.gfile.Exists(settings["summaries_dir"]):
        tf.gfile.MakeDirs(settings["summaries_dir"])
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(settings["summaries_dir"])
            tf.gfile.MakeDirs(settings["summaries_dir"])

def run(settings):
    recreate_subdirectory_structure(settings)
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


def test_hypertune():
    if tf.gfile.Exists(FLAGS.results_test_file):
        os.remove(FLAGS.results_test_file)

    with open(FLAGS.results_val_file, "r") as f:
        content = f.readlines()
        lines = [line.rstrip('\n') for line in content]

        games = []
        lrs = []
        gammas = []
        mean_regrets = []
        mean_nb_subopt_armss = []
        for line in lines:
            results = line.split(" ")
            results = results[1:]
            game, lr, gamma, mean_regret, mean_nb_subopt_arms = [r.split("=")[1] for r in results]
            games.append(game)
            lrs.append(lr)
            gammas.append(gamma)
            mean_regrets.append(mean_regret)
            mean_nb_subopt_armss.append(mean_nb_subopt_arms)
        indices_best_n = np.asarray(mean_regrets).argsort()[-FLAGS.top:][::-1]
        best_lrs = [lrs[i] for i in indices_best_n]
        best_gammas = [gammas[i] for i in indices_best_n]
        best_game = games[0]

        test_envs = TwoArms.get_envs(best_game, FLAGS.nb_test_episodes)

        for i in range(len(indices_best_n)):
            model_name = "best_{}__lr_{}__gamma_{}".format(best_game, best_lrs[i], best_gammas[i])
            load_from_model_name = "d_{}__lr_{}__gamma_{}".format(best_game, best_lrs[i], best_gammas[i])
            print(model_name)
            checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_name)
            summaries_dir = os.path.join(FLAGS.summaries_dir, model_name)
            frames_dir = os.path.join(FLAGS.frames_dir, model_name)

            settings = {"lr": best_lrs[i],
                        "gamma": best_gammas[i],
                        "game": best_game,
                        "model_name": model_name,
                        "checkpoint_dir": checkpoint_dir,
                        "summaries_dir": summaries_dir,
                        "frames_dir": frames_dir,
                        "load_from": os.path.join(FLAGS.checkpoint_dir, load_from_model_name),
                        "envs": test_envs}

            run(settings)


if __name__ == '__main__':
    test_hypertune()