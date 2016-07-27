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

    self.final_probability_of_random_action = sample_final_epsilon()

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
        if self.probability_of_random_action > config.FINAL_RANDOM_ACTION_PROB:
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
          print 'EPISODE: ', T, ' time thread: ', self.time_step, ' result: ', score
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


  def set_feedback(self, obs, action, reward, done):
    obs_resized_grayscaled = cv2.cvtColor(cv2.resize(obs, (config.RESIZED_SCREEN_X, config.RESIZED_SCREEN_Y)),
                                                  cv2.COLOR_BGR2GRAY)
    # set the pixels to all be 0. or 1.
    _, obs_resized_binary = cv2.threshold(obs_resized_grayscaled, 1, 255, cv2.THRESH_BINARY)

    obs_resized_binary = np.reshape(obs_resized_binary, (config.RESIZED_SCREEN_X, config.RESIZED_SCREEN_Y, 1))

    next_state = np.append(self.state[:, :, 1:], obs_resized_binary, axis=2)

    self.replay_memory.append((self.state, action, reward, next_state, done))

    self.state = next_state
    self.time_step += 1

    if len(self.replay_memory) > config.MEMORY_SIZE:
      self.replay_memory.popleft()

    # Store transitions to replay start size then start training
    if self.time_step > config.OBSERVATION_STEPS:
      self.train()

    if self.time_step % config.SAVE_EVERY_X_STEPS == 0:
      self.network.save_network(self.time_step)

    # gradually reduce the probability of a random actionself.
    if self.probability_of_random_action > config.FINAL_RANDOM_ACTION_PROB and len(self.replay_memory) > config.OBSERVATION_STEPS:
      self.probability_of_random_action -= (config.INITIAL_RANDOM_ACTION_PROB - config.FINAL_RANDOM_ACTION_PROB) / config.EXPLORE_STEPS


  def train(self):
    # sample a mini_batch to train on
    minibatch = random.sample(self.replay_memory, config.MINI_BATCH_SIZE)
    # get the batch variables
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    one_hot_actions = np.eye(self.env.action_space.n)[action_batch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    y_batch = []
    # this gives us the agents expected reward for each action we might
    q_value_batch = self.network.get_target_q_batch(next_state_batch)

    for i in range(0, config.MINI_BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        y_batch.append(reward_batch[i])
      else:
        y_batch.append(reward_batch[i] + config.FUTURE_REWARD_DISCOUNT * np.max(q_value_batch[i]))
    # y_batch = np.array(y_batch)
    # y_batch = np.reshape(y_batch, [len(y_batch), 1])

    # learn that these actions in these states lead to this reward
    self.network.train(y_batch, state_batch, one_hot_actions)

    # self.network.save_network(time_step=self.time_step)


