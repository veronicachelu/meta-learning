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

def run_baseline():
    test_envs = TwoArms.get_envs(FLAGS.game, FLAGS.nb_test_episodes)

    model_name = "baseline"

    settings = {"model_name": model_name,
                "game": FLAGS.game,
                "envs": test_envs,
                "exp_type": "evaluate_baseline"}

    tf.reset_default_graph()

    with tf.device("/cpu:0"):
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
            agents.append(RandomAgent(envs[i], i, settings))

    with tf.Session() as sess:
        coord = tf.train.Coordinator()

        agent_threads = []
        for agent in agents:
            agent_play = lambda: agent.play(coord)
            thread = threading.Thread(target=agent_play)
            thread.start()
            agent_threads.append(thread)
        coord.join(agent_threads)


if __name__ == '__main__':
    run_baseline()