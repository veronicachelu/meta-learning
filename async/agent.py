import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from network import AC_Network
from random import choice
from utils import update_target_graph, process_frame, discount
from atari_environment import AtariEnvironment
import flags

FLAGS = tf.app.flags.FLAGS


class Worker():
    def __init__(self, game, name, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/train_" + str(self.number))
        self.summary = tf.Summary()

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(self.name, trainer, self.summary)
        self.update_local_ops = update_target_graph('global', self.name)

        self.actions = np.zeros([FLAGS.nb_actions])
        # End Doom set-up
        self.env = AtariEnvironment(gym_env=game, resized_width=FLAGS.resized_width,
                                    resized_height=FLAGS.resized_height,
                                    agent_history_length=FLAGS.agent_history_length)

    def train(self, rollout, sess, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, FLAGS.gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        td_residuals = rewards + FLAGS.gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(td_residuals, FLAGS.gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.stack(observations, axis=0),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                s = self.env.get_initial_state()
                episode_frames.append(s)
                rnn_state = self.local_AC.state_init

                while d == False:
                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs: [s],
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    s1, r, d, info = self.env.step(a)
                    if FLAGS.show_training:
                        self.env.env.render()
                    r = np.clip(r, -1, 1)

                    if d == False:
                        episode_frames.append(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == FLAGS.max_episode_buffer_size and d != True:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.inputs: [s],
                                                 self.local_AC.state_in[0]: rnn_state[0],
                                                 self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    # if self.name == 'worker_0' and episode_count % 25 == 0:
                    #     time_per_step = 0.05
                    #     images = np.array(episode_frames)
                    #     make_gif(images, './frames/image' + str(episode_count) + '.gif',
                    #              duration=len(images) * time_per_step, true_image=True, salience=False)
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])

                    self.summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    self.summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    self.summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    self.summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    self.summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    self.summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    self.summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    self.summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(self.summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
