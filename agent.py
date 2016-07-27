from collections import deque
import numpy as np
import config
import random
from network import DQNetwork
import cv2
import threading
import time

class AgentAsyncDQN(threading.Thread):

  def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
    threading.Thread.__init__(self, group=group, target=target, name=name,
                                  verbose=verbose)
    self.thread_id, self.env, self.network = args

    self.actions = range(self.env.action_space.n)

    self.probability_of_random_action = config.INITIAL_RANDOM_ACTION_PROB

    self.final_probability_of_random_action = self.sample_final_epsilon()

    self.time_step = 0

    # Initialize network gradients
    self.s_batch = []
    self.a_batch = []
    self.y_batch = []



  def run(self):
    global T

    print "Starting thread ", self.thread_id, "with final epsilon ", self.final_probability_of_random_action

    time.sleep(3 * self.thread_id)

    while T < config.EPISODES:
      obs = self.env.reset()
      self.set_initial_state(obs)

      # Set up per-episode counters
      ep_reward = 0
      episode_ave_max_q = 0
      ep_t = 0

      while True:
        action = self.get_action()
        q_values = self.network.get_q_values(self.state)

        obs, reward, done, _ = self.env.step(action)

        # build next state
        obs_resized_grayscaled = cv2.cvtColor(cv2.resize(obs, (config.RESIZED_SCREEN_X, config.RESIZED_SCREEN_Y)),
                                              cv2.COLOR_BGR2GRAY)
        # set the pixels to all be 0. or 1.
        _, obs_resized_binary = cv2.threshold(obs_resized_grayscaled, 1, 255, cv2.THRESH_BINARY)

        obs_resized_binary = np.reshape(obs_resized_binary, (config.RESIZED_SCREEN_X, config.RESIZED_SCREEN_Y, 1))

        next_state = np.append(self.state[:, :, 1:], obs_resized_binary, axis=2)

        # gradually reduce the probability of a random actionself.
        if self.probability_of_random_action > self.final_probability_of_random_action:
          self.probability_of_random_action -= (config.INITIAL_RANDOM_ACTION_PROB - self.final_probability_of_random_action)\
                                               / config.EXPLORE_STEPS

        # Accumulate gradients
        # this gives us the agents expected reward for each action we might
        target_q_values = self.network.get_target_q_values([next_state])
        clipped_reward = np.clip(reward, -1, 1)

        if done:
          self.y_batch.append(clipped_reward)
        else:
          self.y_batch.append(clipped_reward + config.FUTURE_REWARD_DISCOUNT * np.max(target_q_values))

        self.a_batch.append(action)
        self.s_batch.append(self.state)

        # Update the state and counters
        self.state = next_state
        T += 1
        self.time_step += 1

        ep_t += 1
        ep_reward += reward
        episode_ave_max_q += np.max(q_values)

        # Update target network
        if T % config.TARGET_NETWORK_UPDATE_FREQUENCY == 0:
          self.network.reset_target_network()

        # Update online network
        if self.time_step % config.NETWORK_UPDATE_FREQUENCY == 0 or done:
          if self.s_batch:
            one_hot_actions = np.eye(self.env.action_space.n)[self.a_batch]
            self.network.grad_update(self.y_batch, self.s_batch, one_hot_actions)
          # Clear gradients
          self.s_batch = []
          self.a_batch = []
          self.y_batch = []

        # Save model progress
        if self.time_step % config.SAVE_EVERY_X_STEPS == 0:
          self.network.save_network(self.time_step)

        if done:
          stats = [ep_reward, episode_ave_max_q / float(ep_t), self.probability_of_random_action]
          self.network.update_summaries(stats)

          print "THREAD:", self.thread_id, "/ TIME", T, "/ TIMESTEP", self.time_step, "/ EPSILON", \
            self.probability_of_random_action, "/ REWARD", ep_reward, \
            "/ Q_MAX %.4f" % (episode_ave_max_q / float(ep_t)), "/ EPSILON PROGRESS", self.time_step / float(config.EPISODES)
          break


  def set_initial_state(self, obs):
    obs_resized_grayscaled = cv2.cvtColor(cv2.resize(obs, (config.RESIZED_SCREEN_X, config.RESIZED_SCREEN_Y)),
                                          cv2.COLOR_BGR2GRAY)
    # set the pixels to all be 0. or 1.
    _, obs_resized_binary = cv2.threshold(obs_resized_grayscaled, 1, 255, cv2.THRESH_BINARY)

    # the _last_state will contain the image data from the last self.STATE_FRAMES frames
    self.state = np.stack(tuple(obs_resized_binary for _ in range(config.STATE_FRAMES)), axis=2)


  def get_action(self):
    if (random.random() <= self.probability_of_random_action):
      # choose an action randomly
      action = self.env.action_space.sample()
    else:
      action = self.network.get_action(self.state)

    return action


  def sample_final_epsilon(self):
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1, .01, .5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]


