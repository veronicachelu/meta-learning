import numpy as np
import tensorflow as tf
import random
import random
FLAGS = tf.app.flags.FLAGS


class IntelligentAgent():
    def __init__(self, game):
        self.episode_rewards = []

        self.episode_optimal_rewards = []
        self.episodes_suboptimal_arms = []

        # self.summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/baseline")
        # self.summary = tf.Summary()

        self.env = game

    def get_action_towards_goal(self, info):
        # 0 - up, 1 - down, 2 - left, 3 - right, 4 - 90 counter-clockwise, 5 - 90 clockwise
        # print(info)
        cost = np.abs(info["hero"][0] - info["goal"][0]) + np.abs(info["hero"][1] - info["goal"][1])
        eps = 1e-7
        costs_actions = []
        # action = 0 up
        new_hero_pos = info["hero"]
        if new_hero_pos[0] >= 1:
            new_hero_pos = (new_hero_pos[0] - 1, new_hero_pos[1])
        new_cost = np.abs(new_hero_pos[0] - info["goal"][0]) + np.abs(new_hero_pos[1] - info["goal"][1]) + eps * random.randint(0, 1)
        costs_actions.append(new_cost)

        # action = 1 down
        new_hero_pos = info["hero"]
        if new_hero_pos[0] <= info["grid"][0]:
            new_hero_pos = (new_hero_pos[0] + 1, new_hero_pos[1])
        new_cost = np.abs(new_hero_pos[0] - info["goal"][0]) + np.abs(new_hero_pos[1] - info["goal"][1]) + eps * random.randint(0, 1)
        costs_actions.append(new_cost)

        # action = 2 left
        new_hero_pos = info["hero"]
        if new_hero_pos[1] >= 1:
            new_hero_pos = (new_hero_pos[0], new_hero_pos[1] - 1)
        new_cost = np.abs(new_hero_pos[0] - info["goal"][0]) + np.abs(new_hero_pos[1] - info["goal"][1]) + eps * random.randint(0, 1)
        costs_actions.append(new_cost)

        # action = 3 right
        new_hero_pos = info["hero"]
        if new_hero_pos[1] <= info["grid"][1]:
            new_hero_pos = (new_hero_pos[0], new_hero_pos[1] + 1)
        new_cost = np.abs(new_hero_pos[0] - info["goal"][0]) + np.abs(new_hero_pos[1] - info["goal"][1]) + eps * random.randint(0, 1)
        costs_actions.append(new_cost)

        a = np.argmin(costs_actions)

        return a

    def play(self):

        total_steps = 0
        test_episode_count = 0

        print("Starting baseline agent")
        while True:
            episode_reward = 0
            episode_step_count = 0
            d = False
            _, _, _, info = self.env.reset()
            
            while not d:
                a = self.get_action_towards_goal(info)
                _, r, d, info = self.env.step(a)

                episode_reward += r
                total_steps += 1
                episode_step_count += 1

                # self.env.render()

                if episode_step_count >= 199:
                     d = True

            self.episode_rewards.append(episode_reward)

            if test_episode_count == FLAGS.nb_test_episodes - 1:
                mean_reward = np.mean(self.episode_rewards)

                print("Mean reward for the baseline model is {}".format(mean_reward))
                return 1

            test_episode_count += 1
