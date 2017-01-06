from collections import deque
import tensorflow as tf
import numpy as np
import flags
import random
import cv2
import threading
import time

T = 0

FLAGS = tf.app.flags.FLAGS


class AgentAsyncAC3(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name)
        self.thread_id, self.env, self.network = args

        self.actions = range(self.env.action_space.n)

        self.time_step_thread = 0

    def run(self):
        print("Starting thread ", self.thread_id)

        time.sleep(3 * self.thread_id)
        last_summary_time = 0

        while self.network.T < FLAGS.EPISODES:
            obs = self.env.reset()
            self.set_initial_state(obs)

            # Set up per-episode counters
            ep_reward = 0
            episode_avg_v = 0
            ep_t = 0

            done = False
            self.time_step = 0
            self.time_step_start = self.time_step

            self.s_batch = []
            self.a_batch = []
            self.r_batch = []

            while not (done or ((self.time_step - self.time_step_start) == FLAGS.MAX_TIME_STEPS)):

                policy_output_values = self.network.get_policy_output(self.state)

                action = self.get_action(policy_output_values)

                if self.time_step_thread % FLAGS.PRINT_STATISTICS_EVERY_X_STEPS == 0:
                    print("Thead step: ", self.time_step_thread, "Policy probability max, ",
                          np.max(policy_output_values), \
                          "V ", self.network.get_value_output(self.state)[0])

                obs, reward, done, _ = self.env.step(action)

                # build next state
                obs_resized_grayscaled = cv2.cvtColor(
                    cv2.resize(obs, (FLAGS.RESIZED_SCREEN_X, FLAGS.RESIZED_SCREEN_Y)),
                    cv2.COLOR_BGR2GRAY)
                # set the pixels to all be 0. or 1.
                _, obs_resized_binary = cv2.threshold(obs_resized_grayscaled, 1, 255, cv2.THRESH_BINARY)

                obs_resized_binary = np.reshape(obs_resized_binary,
                                                (FLAGS.RESIZED_SCREEN_X, FLAGS.RESIZED_SCREEN_Y, 1))

                next_state = np.append(self.state[:, :, 1:], obs_resized_binary, axis=2)

                clipped_reward = np.clip(reward, -1, 1)

                one_hot_action = np.eye(self.env.action_space.n)[action]
                self.a_batch.append(one_hot_action)
                self.s_batch.append(self.state)
                self.r_batch.append(clipped_reward)

                # Update the state and counters
                self.state = next_state
                self.network.T += 1
                self.time_step += 1
                self.time_step_thread += 1

                ep_t += 1
                ep_reward += reward

            if done:
                self.v_t = 0
            else:
                self.v_t = self.network.get_value_output(self.state)

            self.r_input = np.zeros(self.time_step)
            for i in reversed(range(self.time_step_start, self.time_step)):
                self.v_t = self.r_batch[i] + FLAGS.FUTURE_REWARD_DISCOUNT * self.v_t
                self.r_input[i] = self.v_t
                episode_avg_v += self.v_t

            last_summary_time = self.network.train(self.s_batch, self.a_batch, self.r_input, last_summary_time)

            # Save model progress
            if self.network.T % FLAGS.SAVE_EVERY_X_STEPS == 0:
                self.network.save_network(self.time_step)

            # if done:
            stats = [ep_reward, episode_avg_v / float(ep_t)]
            self.network.update_summaries(stats)

            print("THREAD:", self.thread_id, "/ TIME", self.network.T, "/ TIMESTEP", self.time_step_thread,
                  "/ REWARD", ep_reward, \
                  "/ V %.4f" % (episode_avg_v / float(ep_t)))

    def set_initial_state(self, obs):
        obs_resized_grayscaled = cv2.cvtColor(cv2.resize(obs, (FLAGS.RESIZED_SCREEN_X, FLAGS.RESIZED_SCREEN_Y)),
                                              cv2.COLOR_BGR2GRAY)
        # set the pixels to all be 0. or 1.
        _, obs_resized_binary = cv2.threshold(obs_resized_grayscaled, 1, 255, cv2.THRESH_BINARY)

        # the _last_state will contain the image data from the last self.STATE_FRAMES frames
        self.state = np.stack(tuple(obs_resized_binary for _ in range(FLAGS.STATE_FRAMES)), axis=2)

    def get_action(self, policy_proba):
        policy_proba = policy_proba - np.finfo(np.float32).epsneg

        histogram = np.random.multinomial(1, policy_proba)
        action_index = int(np.nonzero(histogram)[0])
        return action_index

    def sample_final_epsilon(self):
        """
        Sample a final epsilon value to anneal towards from a distribution.
        These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
        """
        final_epsilons = np.array([.1, .01, .5])
        probabilities = np.array([0.4, 0.3, 0.3])
        return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]
