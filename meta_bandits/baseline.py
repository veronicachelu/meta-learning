import numpy as np
import tensorflow as tf
from network import AC_Network
from utils import update_target_graph, discount, set_image_bandit, set_image_bandit_11_arms, make_gif
import random
FLAGS = tf.app.flags.FLAGS


class RandomAgent():
    def __init__(self, game, thread_id, settings):
        self.name = "random_agent_" + str(thread_id)
        self.thread_id = thread_id
        self.settings = settings
        self.episode_rewards = []

        self.episode_optimal_rewards = []
        self.episodes_suboptimal_arms = []

        self.env = game

    def play(self, coord):

        total_steps = 0
        test_episode_count = 0

        print("Starting agent " + str(self.thread_id))
        while not coord.should_stop():

            episode_rewards_for_optimal_arm = 0
            episode_suboptimal_arm = 0
            episode_reward = [0 for _ in range(FLAGS.nb_actions)]
            episode_step_count = 0
            d = False
            r = 0
            a = 0
            t = 0
            self.env.set(self.settings["envs"][test_episode_count])

            while not d:
                a = random.randint(0, FLAGS.nb_actions)
                r, d, t = self.env.pull_arm(a)

                episode_rewards_for_optimal_arm += self.env.pull_arm_for_test()
                optimal_action = self.env.get_optimal_arm()
                if optimal_action != a:
                    episode_suboptimal_arm += 1

                episode_reward[a] += r
                total_steps += 1
                episode_step_count += 1

            self.episode_rewards.append(np.sum(episode_reward))

            self.episodes_suboptimal_arms.append(episode_suboptimal_arm)
            self.episode_optimal_rewards.append(episode_rewards_for_optimal_arm)

            if test_episode_count == FLAGS.nb_test_episodes - 1:
                episode_regret = [max(o - r, 0) for (o, r) in
                                  zip(self.episode_optimal_rewards, self.episode_rewards)]
                mean_regret = np.mean(episode_regret)
                mean_nb_suboptimal_arms = np.mean(self.episodes_suboptimal_arms)

                print("Mean regret for the model is {}".format(mean_regret))
                print("Regret in terms of suboptimal arms is {}".format(mean_nb_suboptimal_arms))
                return 1

            test_episode_count += 1
