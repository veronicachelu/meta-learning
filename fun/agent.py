import numpy as np
import tensorflow as tf
from network import FUNNetwork
from utils import update_target_graph, discount, set_image_bandit, set_image_bandit_11_arms, make_gif
import os
FLAGS = tf.app.flags.FLAGS


class Agent():
    def __init__(self, game, thread_id, optimizer, global_step):
        self.name = "agent_" + str(thread_id)
        self.thread_id = thread_id
        self.model_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name)
        self.optimizer = optimizer
        self.global_episode = global_step
        self.increment_global_episode = self.global_episode.assign_add(1)
        self.episode_rewards = []

        # if not FLAGS.train:
        self.episode_optimal_rewards = []
        self.episodes_suboptimal_arms = []

        self.episode_lengths = []
        self.episode_mean_w_values = []
        self.episode_mean_mean_intr_reward = []
        self.episode_mean_m_values = []
        self.summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir, FLAGS.model_name) + "/agent_" + str(self.thread_id))
        self.summary = tf.Summary()

        self.local_AC = FUNNetwork(self.name, optimizer, self.global_episode)

        self.update_local_vars = update_target_graph('global', self.name)
        self.env = game

    def train(self, rollout, sess, bootstrap_value, summaries=False):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        timesteps = rollout[:, 3]
        w_values = rollout[:, 5]
        m_values = rollout[:, 6]

        # if FLAGS.meta:
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()

        # The advantage function uses "Generalized Advantage Estimation"
        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        w_discounted_rewards = discount(rewards_plus, FLAGS.w_gamma)[:-1]
        m_discounted_rewards = discount(rewards_plus, FLAGS.m_gamma)[:-1]
        # w_value_plus = np.asarray(w_values.tolist() + [bootstrap_value])
        # m_value_plus = np.asarray(m_values.tolist() + [bootstrap_value])

        w_rnn_state = self.local_AC.w_state_init
        m_rnn_state = self.local_AC.m_state_init
        feed_dict = {self.local_AC.w_extrinsic_return: w_discounted_rewards,
                     self.local_AC.m_extrinsic_return: m_discounted_rewards,
                     self.local_AC.inputs: np.stack(observations, axis=0),
                     self.local_AC.prev_rewards: prev_rewards,
                     self.local_AC.prev_actions: prev_actions,
                     self.local_AC.actions: actions,
                     self.local_AC.w_state_in[0]: w_rnn_state[0],
                     self.local_AC.w_state_in[1]: w_rnn_state[1],
                     self.local_AC.m_state_in[0]: m_rnn_state[0],
                     self.local_AC.m_state_in[1]: m_rnn_state[1]
                     }

        if summaries:
            l, w_v_l, i_r, m_v_l, p_l, g_l, e_l, g_n, v_n, _, ms, img_summ = sess.run([self.local_AC.loss,
                                                                    self.local_AC.w_value_loss,
                                                                    self.local_AC.intr_rewards,
                                                                    self.local_AC.m_value_loss,
                                                                    self.local_AC.w_policy_loss,
                                                                    self.local_AC.goals_loss,
                                                                    self.local_AC.entropy,
                                                                    self.local_AC.grad_norms,
                                                                    self.local_AC.var_norms,
                                                                    self.local_AC.apply_grads,
                                                                    self.local_AC.merged_summary,
                                                                    self.local_AC.image_summaries],
                                                                   feed_dict=feed_dict)
            return l / len(rollout), w_v_l / len(rollout), i_r / len(rollout), m_v_l / len(rollout), \
                   p_l / len(rollout), g_l / len(rollout),\
                   e_l / len(rollout), g_n, v_n, ms, img_summ
        else:
            _ = sess.run([self.local_AC.apply_grads], feed_dict=feed_dict)
            return None

    def play(self, sess, coord, saver):
        episode_count = sess.run(self.global_episode)

        if not FLAGS.train:
            test_episode_count = 0

        total_steps = 0

        print("Starting agent thread " + str(self.thread_id))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                if FLAGS.train and episode_count > FLAGS.max_nb_episodes_train:
                    return 0

                sess.run(self.update_local_vars)
                episode_buffer = []

                episode_w_values = []
                episode_intr_reward = []
                episode_m_values = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                t = 0
                r = 0
                a = 0

                s, _, _, _ = self.env.reset()
                m_rnn_state = self.local_AC.m_state_init
                w_rnn_state = self.local_AC.w_state_init

                while not d:

                    feed_dict = {
                        self.local_AC.inputs: [s],
                        self.local_AC.prev_rewards: [r],
                        self.local_AC.prev_actions: [a],
                        self.local_AC.w_state_in[0]: w_rnn_state[0],
                        self.local_AC.w_state_in[1]: w_rnn_state[1],
                        self.local_AC.m_state_in[0]: m_rnn_state[0],
                        self.local_AC.m_state_in[1]: m_rnn_state[1]
                    }


                    pi, w_v, i_r, m_v, w_rnn_state_new, m_rnn_state_new = sess.run(
                        [self.local_AC.w_policy, self.local_AC.w_value, self.local_AC.intr_rewards, self.local_AC.m_value, self.local_AC.w_state_out,
                         self.local_AC.m_state_out], feed_dict=feed_dict)
                    a = np.random.choice(pi[0], p=pi[0])
                    a = np.argmax(pi == a)

                    w_rnn_state = w_rnn_state_new
                    m_rnn_state = m_rnn_state_new

                    s1, r, d, _ = self.env.step(a)

                    episode_buffer.append([s, a, r, t, d, w_v[0, 0], m_v[0, 0]])
                    episode_w_values.append(w_v[0, 0])
                    episode_intr_reward.append(i_r[0])
                    episode_m_values.append(m_v[0, 0])
                    episode_reward += r
                    total_steps += 1
                    t += 1
                    episode_step_count += 1

                    s = s1

                    # print(t)
                    if t >= FLAGS.BTT_length:
                        d = True

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_w_values.append(np.mean(episode_w_values))
                self.episode_mean_mean_intr_reward.append(np.mean(episode_w_values))
                self.episode_mean_m_values.append(np.mean(episode_m_values))

                if len(episode_buffer) != 0 and FLAGS.train == True:
                    if episode_count % FLAGS.summary_interval == 0 and episode_count != 0:
                        l, w_v_l, i_ret, m_v_l, p_l, g_l, e_l, g_n, v_n, ms, img_sum = self.train(episode_buffer, sess, 0.0, summaries=True)
                    else:
                        self.train(episode_buffer, sess, 0.0)

                if not FLAGS.train and test_episode_count == FLAGS.nb_test_episodes - 1:
                    print("Mean reward for the model is {}".format(np.mean(self.episode_rewards)))
                    return 1

                if FLAGS.train and episode_count % FLAGS.summary_interval == 0 and episode_count != 0 and \
                                self.name == 'agent_0':
                    if episode_count % FLAGS.checkpoint_interval == 0 and self.name == 'agent_0' and FLAGS.train == True:
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk',
                                   global_step=self.global_episode)
                        print("Saved Model at {}".format(self.model_path + '/model-' + str(episode_count) + '.cptk'))

                    mean_reward = np.mean(self.episode_rewards[-FLAGS.summary_interval:])
                    mean_length = np.mean(self.episode_lengths[-FLAGS.summary_interval:])
                    mean_w_value = np.mean(self.episode_mean_w_values[-FLAGS.summary_interval:])
                    mean_intr_reward = np.mean(self.episode_mean_mean_intr_reward[-FLAGS.summary_interval:])
                    mean_m_value = np.mean(self.episode_mean_m_values[-FLAGS.summary_interval:])

                    self.summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))

                    self.summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    self.summary.value.add(tag='Perf/W_Value', simple_value=float(mean_w_value))
                    self.summary.value.add(tag='Perf/Intrinsic_reward', simple_value=float(mean_intr_reward))
                    self.summary.value.add(tag='Perf/M_Value', simple_value=float(mean_m_value))


                    if FLAGS.train:
                        self.summary.value.add(tag='Losses/Total Loss', simple_value=float(l))
                        self.summary.value.add(tag='Losses/W_Value Loss', simple_value=float(w_v_l))
                        self.summary.value.add(tag='Losses/M_Value Loss', simple_value=float(m_v_l))
                        self.summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        self.summary.value.add(tag='Losses/Goal Loss', simple_value=float(g_l))
                        self.summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        self.summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        self.summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
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
                    for s in img_sum:
                        self.summary_writer.add_summary(s, episode_count)
                    self.summary_writer.add_summary(self.summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'agent_0':
                    sess.run(self.increment_global_episode)
                if not FLAGS.train:
                    test_episode_count += 1
                episode_count += 1
