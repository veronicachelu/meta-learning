import numpy as np
import tensorflow as tf
from utils import set_image_bandit, make_gif
import random
FLAGS = tf.app.flags.FLAGS


class RandomAgent():
    def __init__(self, game, thread_id, settings):
        self.name = "random_agent_" + str(thread_id)
        self.thread_id = thread_id
        self.settings = settings
        self.episode_rewards = []

        self.episode_regrets = []
        self.episodes_suboptimal_arms = []

        self.env = game

    def play(self, coord):

        total_steps = 0
        test_episode_count = 0

        print("Starting agent " + str(self.thread_id))
        while not coord.should_stop():

            episode_regret = 0
            episode_suboptimal_arm = 0
            episode_reward = [0 for _ in range(FLAGS.nb_actions)]
            episode_step_count = 0
            episode_frames = []
            d = False
            r = 0
            a = 0
            t = 0
            self.env.set(self.settings["envs"][test_episode_count])

            while not d:
                a = random.randint(0, FLAGS.nb_actions - 1)
                r, d, t = self.env.pull_arm(a)

                episode_frames.append(set_image_bandit(episode_reward, self.env.get_bandit(), a, t))

                episode_regret += self.env.get_timestep_regret(a)
                optimal_action = self.env.get_optimal_arm()
                if optimal_action != a:
                    episode_suboptimal_arm += 1

                episode_reward[a] += r
                total_steps += 1
                episode_step_count += 1

            self.episode_rewards.append(np.sum(episode_reward))

            self.episodes_suboptimal_arms.append(episode_suboptimal_arm)
            self.episode_regrets.append(episode_regret)

            self.images = np.array(episode_frames)
            make_gif(self.images, FLAGS.frames_test_dir + '/image' + str(test_episode_count) + '.gif',
                     duration=len(self.images) * 0.1, true_image=True)

            if test_episode_count == FLAGS.nb_test_episodes - 1:
                mean_regret = np.mean(self.episode_regrets)
                mean_nb_suboptimal_arms = np.mean(self.episodes_suboptimal_arms)

                print("Mean regret for the model is {}".format(mean_regret))
                print("Regret in terms of suboptimal arms is {}".format(mean_nb_suboptimal_arms))
                return 1

            test_episode_count += 1
