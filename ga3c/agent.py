from multiprocessing import Process, Queue, Value
import numpy as np
import flags
import gym
import gym_fast_envs
import tensorflow as tf
from atari_environment import AtariEnvironment
import time
from scipy.signal import lfilter
FLAGS = tf.app.flags.FLAGS

class Agent(Process):
    def __init__(self, id, prediction_q, training_q):
        super(Agent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q

        gym_env = gym.make(FLAGS.game)
        gym_env.seed(FLAGS.seed)

        self.env = AtariEnvironment(gym_env=gym_env, resized_width=FLAGS.resized_width,
                                    resized_height=FLAGS.resized_height,
                                    agent_history_length=FLAGS.agent_history_length)

        self.nb_actions = len(self.env.gym_actions)
        self.wait_q = Queue(maxsize=1)
        self.stop = Value('i', 0)

    def run(self):
        time.sleep(np.random.rand())

        while not self.stop.value:
            # total_reward = 0
            # total_length = 0
            for episode_buffer in self.run_episode_generator():
                self.training_q.put(episode_buffer)

    def run_episode_generator(self):
        s, _ = self.env.get_initial_state()

        d = False
        episode_buffer = []
        episode_reward = 0
        episode_step_count = 0

        while not d:
            self.prediction_q.put((self.id, s))
            pi, v = self.wait_q.get()
            a = np.random.choice(pi[0], p=pi[0])
            a = np.argmax(pi == a)

            s1, r, d, info = self.env.step(a)

            r = np.clip(r, -1, 1)

            episode_buffer.append([s, a, pi, r, s1, d, v[0, 0]])
            episode_reward += r
            episode_step_count += 1
            s = s1

            if len(episode_buffer) == FLAGS.max_episode_buffer_size and not d:
                self.prediction_q.put((self.id, s))
                pi, v1 = self.wait_q.get()
                updated_episode_buffer = self.get_training_data(episode_buffer, v1)
                yield updated_episode_buffer
            if d:
                break

        if len(episode_buffer) != 0:
            updated_episode_buffer = self.get_training_data(episode_buffer, 0)
            yield updated_episode_buffer

    def discount(self, x):
        return lfilter([1], [1, -FLAGS.gamma], x[::-1], axis=0)[::-1]

    def get_training_data(self, rollout, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        pis = rollout[:, 2]
        rewards = rollout[:, 3]
        next_observations = rollout[:, 4]
        values = rollout[:, 5]

        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = self.discount(rewards_plus, FLAGS.gamma)[:-1]
        value_plus = np.asarray(values.tolist() + [bootstrap_value])
        policy_target = discounted_rewards - value_plus[:-1]

        rollout.extend([discounted_rewards])
