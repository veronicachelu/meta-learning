import tensorflow as tf
import numpy as np
from utils import update_target_graph, discount, set_image_bandit, make_gif
from network import AC_Network
import flags
from threading import Thread, Lock
import operator
FLAGS = tf.app.flags.FLAGS

class Worker():
    def __init__(self, game, name, trainer, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = FLAGS.checkpoint_dir
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []

        #if not FLAGS.train:
        self.episode_optimal_rewards = []
        self.episodes_suboptimal_arms = []

        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = game

    def train(self, rollout, sess, bootstrap_value):
        rollout = np.array(rollout)
        actions = rollout[:, 0]
        rewards = rollout[:, 1]
        timesteps = rollout[:, 2]
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        values = rollout[:, 4]

        self.pr = prev_rewards
        self.pa = prev_actions
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
                     self.local_AC.prev_rewards: np.vstack(prev_rewards),
                     self.local_AC.prev_actions: prev_actions,
                     self.local_AC.actions: actions,
                     self.local_AC.timestep: np.vstack(timesteps),
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
        if not FLAGS.train:
            test_episode_count = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                if not FLAGS.train:
                    print("Episode {}".format(test_episode_count))
                episode_rewards_for_optimal_arm = 0
                episode_suboptimal_arm = 0
                episode_values = []
                episode_frames = []
                episode_reward = [0 for i in range(FLAGS.nb_actions)]
                episode_step_count = 0
                d = False
                r = 0
                a = 0
                t = 0
                self.env.reset()
                rnn_state = self.local_AC.state_init

                while d == False:
                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state_new = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={
                            self.local_AC.prev_rewards: [[r]],
                            self.local_AC.timestep: [[t]],
                            self.local_AC.prev_actions: [a],
                            self.local_AC.state_in[0]: rnn_state[0],
                            self.local_AC.state_in[1]: rnn_state[1]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    rnn_state = rnn_state_new
                    r, d, t = self.env.pullArm(a)
                    #if not FLAGS.train:
                    episode_rewards_for_optimal_arm += self.env.pullArmForTest()
                    optimal_action = self.env.get_optimal_arm()
                    if optimal_action != a:
                        episode_suboptimal_arm += 1
                    episode_buffer.append([a, r, t, d, v[0, 0]])
                    episode_values.append(v[0, 0])
                    if not FLAGS.game == '11arms':
                        episode_frames.append(set_image_bandit(episode_reward, self.env.get_bandit(), a, t))
                    else:
                        episode_frames.append(set_image_bandit(episode_reward, self.env.get_optimal_arm(), a, t))
                    episode_reward[a] += r
                    total_steps += 1
                    episode_step_count += 1

                self.episode_rewards.append(np.sum(episode_reward))

                self.episodes_suboptimal_arms.append(episode_suboptimal_arm)
                self.episode_optimal_rewards.append(episode_rewards_for_optimal_arm)
                if not FLAGS.train:
                    print("Episode total reward was: {} vs optimal reward {}".format(np.sum(episode_reward), episode_rewards_for_optimal_arm))
                    print("Regret is {}".format(max(episode_rewards_for_optimal_arm - np.sum(episode_reward), 0)))
                    print("Suboptimal arms in the episode: {}".format(episode_suboptimal_arm))

                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0 and FLAGS.train == True:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, 0.0)

                if episode_count % FLAGS.nb_test_episodes:
                    episode_regret = [max(o - r, 0) for (o, r) in
                                      zip(self.episode_optimal_rewards[-150:], self.episode_rewards[-150:])]
                    mean_regret = np.mean(episode_regret)
                    mean_nb_suboptimal_arms = np.mean(self.episodes_suboptimal_arms[-150:])

                if not FLAGS.train and test_episode_count == FLAGS.nb_test_episodes:
                    print("Mean regret for the model is {}".format(mean_regret))
                    print("Regret in terms of suboptimal arms is {}".format(mean_nb_suboptimal_arms))
                    return 1

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if FLAGS.train and episode_count % 50 == 0 and episode_count != 0:
                    if episode_count % FLAGS.checkpoint_interval == 0 and self.name == 'worker_0' and FLAGS.train == True:
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")

                    if episode_count % FLAGS.frames_interval == 0 and self.name == 'worker_0':
                        self.images = np.array(episode_frames)
                        make_gif(self.images, './frames/image' + str(episode_count) + '.gif',
                                 duration=len(self.images) * 0.1, true_image=True)

                    mean_reward = np.mean(self.episode_rewards[-50:])
                    mean_length = np.mean(self.episode_lengths[-50:])
                    mean_value = np.mean(self.episode_mean_values[-50:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))

                    if FLAGS.train == True:
                        if episode_count % FLAGS.nb_test_episodes:
                            summary.value.add(tag='Mean Regret', simple_value=float(mean_regret))
                            summary.value.add(tag='Mean NSuboptArms', simple_value=float(mean_nb_suboptimal_arms))
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                if not FLAGS.train:
                    test_episode_count += 1
                episode_count += 1