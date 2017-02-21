from threading import Lock

import numpy as np
import tensorflow as tf
from network import ACNetwork
from network_lstm import ACNetworkLSTM

from utils import update_target_graph, discount
import flags

FLAGS = tf.app.flags.FLAGS
# Starting threads
main_lock = Lock()


class Worker():
    def __init__(self, game, sess, thread_id, nb_actions, optimizer, global_step):
        self.name = "worker_" + str(thread_id)
        self.thread_id = thread_id
        self.model_path = FLAGS.checkpoint_dir
        self.trainer = optimizer
        self.global_episode = global_step
        self.increment_global_episode = self.global_episode.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        self.sess = sess
        self.graph = sess.graph
        # self.summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/worker_" + str(self.thread_id), self.graph)
        self.summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/worker_" + str(self.thread_id))
        self.summary = tf.Summary()

        if FLAGS.lstm:
            self.local_AC = ACNetworkLSTM(self.name, nb_actions, optimizer)
        else:
            self.local_AC = ACNetwork(self.name, nb_actions, optimizer)

        self.update_local_ops = update_target_graph('global', self.name)

        self.actions = np.zeros([nb_actions])
        self.env = game

    def train(self, rollout, bootstrap_value, summaries=False):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        if FLAGS.meta:
            prev_rewards = [0] + rewards[:-1].tolist()
            prev_actions = [0] + actions[:-1].tolist()
        values = rollout[:, 5]

        # The advantage function uses "Generalized Advantage Estimation"
        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(rewards_plus, FLAGS.gamma)[:-1]
        value_plus = np.asarray(values.tolist() + [bootstrap_value])
        policy_target = discounted_rewards - value_plus[:-1]
        if FLAGS.gen_adv:
            td_residuals = rewards + FLAGS.gamma * value_plus[1:] - value_plus[:-1]
            advantages = discount(td_residuals, FLAGS.gamma)
            policy_target = advantages

        if FLAGS.lstm:
            if FLAGS.meta:
                rnn_state = self.local_AC.state_init
                feed_dict = {self.local_AC.target_v: discounted_rewards,
                             self.local_AC.prev_rewards: np.vstack(prev_rewards),
                             self.local_AC.prev_actions: prev_actions,
                             self.local_AC.actions: actions,
                             self.local_AC.inputs: np.stack(observations, axis=0),
                             self.local_AC.advantages: policy_target,
                             self.local_AC.state_in[0]: rnn_state[0],
                             self.local_AC.state_in[1]: rnn_state[1]}
            else:
                rnn_state = self.local_AC.state_init
                feed_dict = {self.local_AC.target_v: discounted_rewards,
                             self.local_AC.inputs: np.stack(observations, axis=0),
                             self.local_AC.actions: actions,
                             self.local_AC.advantages: policy_target,
                             self.local_AC.state_in[0]: rnn_state[0],
                             self.local_AC.state_in[1]: rnn_state[1]}
        else:
            feed_dict = {self.local_AC.target_v: discounted_rewards,
                         self.local_AC.inputs: np.stack(observations, axis=0),
                         self.local_AC.actions: actions,
                         self.local_AC.advantages: policy_target}
        if summaries:
            l, v_l, p_l, e_l, g_n, v_n, _, ms, max_v, min_v, mean_v, max_r, min_r, mean_r = self.sess.run(
                [self.local_AC.loss,
                 self.local_AC.value_loss,
                 self.local_AC.policy_loss,
                 self.local_AC.entropy,
                 self.local_AC.grad_norms,
                 self.local_AC.var_norms,
                 self.local_AC.apply_grads,
                 self.local_AC.merged_summary,
                 self.local_AC.max_value,
                 self.local_AC.min_value,
                 self.local_AC.mean_value,
                 self.local_AC.max_reward,
                 self.local_AC.min_reward,
                 self.local_AC.mean_reward],
                feed_dict=feed_dict)
            return l / len(rollout), v_l / len(rollout), p_l / len(rollout), e_l / len(
                rollout), g_n, v_n, ms, max_v, min_v, mean_v, max_r, min_r, mean_r
        else:
            _ = self.sess.run([self.local_AC.apply_grads], feed_dict=feed_dict)
        return None

    def play(self, coord, saver):
        episode_count = self.sess.run(self.global_episode)
        total_steps = 0

        print("Starting worker " + str(self.thread_id))
        with self.sess.as_default(), self.graph.as_default():
            while not coord.should_stop():

                self.sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                if FLAGS.meta:
                    r = 0
                    a = 0

                s = self.env.get_initial_state()
                episode_frames.append(s)
                if FLAGS.lstm:
                    rnn_state = self.local_AC.state_init

                while not d:
                    if FLAGS.lstm:
                        if FLAGS.meta:
                            feed_dict = {
                                self.local_AC.prev_rewards: [[r]],
                                self.local_AC.inputs: [[s]],
                                self.local_AC.prev_actions: [a],
                                self.local_AC.state_in[0]: rnn_state[0],
                                self.local_AC.state_in[1]: rnn_state[1]}
                        else:
                            feed_dict = {self.local_AC.inputs: [s],
                                         self.local_AC.state_in[0]: rnn_state[0],
                                         self.local_AC.state_in[1]: rnn_state[1]}

                        pi, v, rnn_state = self.sess.run(
                            [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                            feed_dict=feed_dict)
                    else:
                        feed_dict = {self.local_AC.inputs: [s]}

                        pi, v = self.sess.run(
                            [self.local_AC.policy, self.local_AC.value],
                            feed_dict=feed_dict)

                    a = np.random.choice(pi[0], p=pi[0])
                    a = np.argmax(pi == a)

                    s1, r, d, info = self.env.step(a)

                    r = np.clip(r, -1, 1)

                    if not d:
                        episode_frames.append(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    if len(episode_buffer) == FLAGS.max_episode_buffer_size and not d:
                        if FLAGS.lstm:
                            if FLAGS.meta:
                                feed_dict = {
                                    self.local_AC.prev_rewards: [[r]],
                                    self.local_AC.inputs: [[s]],
                                    self.local_AC.prev_actions: [a],
                                    self.local_AC.state_in[0]: rnn_state[0],
                                    self.local_AC.state_in[1]: rnn_state[1]}
                            else:
                                feed_dict = {self.local_AC.inputs: [s],
                                             self.local_AC.state_in[0]: rnn_state[0],
                                             self.local_AC.state_in[1]: rnn_state[1]}
                            v1 = self.sess.run(self.local_AC.value,
                                               feed_dict=feed_dict)[0, 0]
                        else:
                            feed_dict = {self.local_AC.inputs: [s]}
                            v1 = self.sess.run(self.local_AC.value,
                                               feed_dict=feed_dict)[0, 0]
                        if episode_count % FLAGS.summary_interval == 0 and episode_count != 0:
                            l, v_l, p_l, e_l, g_n, v_n, ms, max_v, min_v, mean_v, max_r, min_r, mean_r = self.train(
                                episode_buffer, v1, summaries=True)
                        else:
                            self.train(episode_buffer, v1)
                        episode_buffer = []
                        self.sess.run(self.update_local_ops)
                    if d:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if len(episode_buffer) != 0:
                    if episode_count % FLAGS.summary_interval == 0 and episode_count != 0:
                        l, v_l, p_l, e_l, g_n, v_n, ms, max_v, min_v, mean_v, max_r, min_r, mean_r = self.train(
                            episode_buffer, 0.0, summaries=True)
                    else:
                        self.train(episode_buffer, 0.0)

                if episode_count % FLAGS.summary_interval == 0 and episode_count != 0:
                    if episode_count % FLAGS.checkpoint_interval == 0 and self.name == 'worker_0' and FLAGS.train == True:
                        saver.save(self.sess, self.model_path + '/model-' + str(episode_count) + '.cptk',
                                   global_step=self.global_episode)
                        print("Saved Model at {}".format(self.model_path + '/model-' + str(episode_count) + '.cptk'))

                    mean_reward = np.mean(self.episode_rewards[-FLAGS.summary_interval:])
                    mean_length = np.mean(self.episode_lengths[-FLAGS.summary_interval:])
                    mean_value = np.mean(self.episode_mean_values[-FLAGS.summary_interval:])

                    if episode_count % FLAGS.test_performance_interval == 0:
                        won_games = self.episode_rewards[-FLAGS.test_performance_interval:].count(1)
                        self.summary.value.add(tag='Perf/Won Games/1000', simple_value=float(won_games))

                    self.summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    self.summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    self.summary.value.add(tag='Perf/Value', simple_value=float(mean_value))

                    # if FLAGS.train:
                    self.summary.value.add(tag='Value/Max', simple_value=float(max_v))
                    self.summary.value.add(tag='Value/Min', simple_value=float(min_v))
                    self.summary.value.add(tag='Value/Mean', simple_value=float(mean_v))
                    self.summary.value.add(tag='Reward/Max', simple_value=float(max_r))
                    self.summary.value.add(tag='Reward/Min', simple_value=float(min_r))
                    self.summary.value.add(tag='Reward/Mean', simple_value=float(mean_r))

                    self.summary.value.add(tag='Losses/Total Loss', simple_value=float(l))
                    self.summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    self.summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    self.summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    self.summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    self.summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    # if False:
                    summaries = tf.Summary().FromString(ms)
                    sub_summaries_dict = {}
                    for value in summaries.value:
                        value_field = value.WhichOneof('value')
                        value_ifo = sub_summaries_dict.setdefault(value.tag,
                                                                  {'value_field': None, 'values': []})
                        if not value_ifo['value_field']:
                            value_ifo['value_field'] = value_field
                        else:
                            assert value_ifo['value_field'] == value_field
                        value_ifo['values'].append(getattr(value, value_field))

                    for name, value_ifo in sub_summaries_dict.items():
                        summary_value = self.summary.value.add()
                        summary_value.tag = name
                        if value_ifo['value_field'] == 'histo':
                            values = value_ifo['values']
                            summary_value.histo.min = min([x.min for x in values])
                            summary_value.histo.max = max([x.max for x in values])
                            summary_value.histo.num = sum([x.num for x in values])
                            summary_value.histo.sum = sum([x.sum for x in values])
                            summary_value.histo.sum_squares = sum([x.sum_squares for x in values])
                            for lim in values[0].bucket_limit:
                                summary_value.histo.bucket_limit.append(lim)
                            for bucket in values[0].bucket:
                                summary_value.histo.bucket.append(bucket)
                        else:
                            print(
                                'Warning: could not aggregate summary of type {}'.format(value_ifo['value_field']))

                    self.summary_writer.add_summary(self.summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    self.sess.run(self.increment_global_episode)
                episode_count += 1
