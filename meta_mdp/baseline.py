import numpy as np
import tensorflow as tf
import random
FLAGS = tf.app.flags.FLAGS


class RandomAgent():
    def __init__(self, game):
        self.episode_rewards = []

        self.episode_optimal_rewards = []
        self.episodes_suboptimal_arms = []

        # self.summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/baseline")
        # self.summary = tf.Summary()

        self.env = game

    def play(self):

        total_steps = 0
        test_episode_count = 0

        print("Starting baseline agent")
        while True:
            episode_reward = 0
            episode_step_count = 0
            d = False
            _ = self.env.reset()

            while not d:
                a = random.randint(0, FLAGS.nb_actions - 1)
                _, r, d, _ = self.env.step(a)

                episode_reward += r
                total_steps += 1
                episode_step_count += 1

                # self.env.render()

                if episode_step_count >= 99:
                    d = True

            self.episode_rewards.append(episode_reward)

            if test_episode_count == FLAGS.nb_test_episodes - 1:
                mean_reward = np.mean(self.episode_rewards)

                print("Mean reward for the baseline model is {}".format(mean_reward))
                return 1

            test_episode_count += 1
