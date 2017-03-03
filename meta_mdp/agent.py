import numpy as np
import tensorflow as tf
from network import ACNetwork
from utils import update_target_graph, discount, set_image_bandit, set_image_bandit_11_arms, make_gif

FLAGS = tf.app.flags.FLAGS


class Agent():
    def __init__(self, game, thread_id, optimizer, global_step):
        self.name = "worker_" + str(thread_id)
        self.thread_id = thread_id
        self.model_path = FLAGS.checkpoint_dir
        self.optimizer = optimizer
        self.global_episode = global_step
        self.increment_global_episode = self.global_episode.assign_add(1)
        self.episode_rewards = []

        # if not FLAGS.train:
        self.episode_optimal_rewards = []
        self.episodes_suboptimal_arms = []

        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/worker_" + str(self.thread_id))
        self.summary = tf.Summary()

        self.local_AC = ACNetwork(self.name, optimizer, self.global_episode)
        self.update_local_vars = update_target_graph('global', self.name)
        self.env = game

    def train(self, rollout, sess, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        timesteps = rollout[:, 3]
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

        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.stack(observations, axis=0),
                     self.local_AC.prev_rewards: np.vstack(prev_rewards),
                     self.local_AC.prev_actions: prev_actions,
                     self.local_AC.actions: actions,
                     self.local_AC.timestep: np.vstack(timesteps),
                     self.local_AC.advantages: policy_target,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}

        l, v_l, p_l, e_l, g_n, v_n, _, ms = sess.run([self.local_AC.loss,
                                                      self.local_AC.value_loss,
                                                      self.local_AC.policy_loss,
                                                      self.local_AC.entropy,
                                                      self.local_AC.grad_norms,
                                                      self.local_AC.var_norms,
                                                      self.local_AC.apply_grads,
                                                      self.local_AC.merged_summary],
                                                     feed_dict=feed_dict)
        return l / len(rollout), v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n, ms

    def play(self, sess, coord, saver):
        episode_count = sess.run(self.global_episode)

        total_steps = 0
        if not FLAGS.train:
            test_episode_count = 0

        print("Starting worker " + str(self.thread_id))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                if FLAGS.train and episode_count > FLAGS.max_nb_episodes_train:
                    return 0

                sess.run(self.update_local_vars)
                episode_buffer = []

                if not FLAGS.train:
                    print("Episode {}".format(test_episode_count))

                episode_rewards_for_optimal_arm = 0
                episode_suboptimal_arm = 0
                episode_values = []
                episode_frames = []
                episode_reward = [0 for _ in range(FLAGS.nb_actions)]
                episode_step_count = 0
                d = False
                r = 0
                a = 0
                t = 0

                s = self.env.reset()
                rnn_state = self.local_AC.state_init

                while not d:
                    feed_dict = {
                        self.local_AC.inputs: [s],
                        self.local_AC.prev_rewards: [[r]],
                        self.local_AC.timestep: [[t]],
                        self.local_AC.prev_actions: [a],
                        self.local_AC.state_in[0]: rnn_state[0],
                        self.local_AC.state_in[1]: rnn_state[1]}

                    pi, v, rnn_state_new = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out], feed_dict=feed_dict)
                    a = np.random.choice(pi[0], p=pi[0])
                    a = np.argmax(pi == a)

                    rnn_state = rnn_state_new
                    s, r, d, _ = self.env.step(a)
                    episode_buffer.append([s, a, r, t, d, v[0, 0]])
                    episode_values.append(v[0, 0])
                    episode_reward[a] += r
                    total_steps += 1
                    t+= 1
                    episode_step_count += 1

                self.episode_rewards.append(np.sum(episode_reward))
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if len(episode_buffer) != 0 and FLAGS.train == True:
                    l, v_l, p_l, e_l, g_n, v_n, ms = self.train(episode_buffer, sess, 0.0)

                if FLAGS.train and episode_count % FLAGS.summary_interval == 0 and episode_count != 0:
                    if episode_count % FLAGS.checkpoint_interval == 0 and self.name == 'worker_0' and FLAGS.train == True:
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk',
                                   global_step=self.global_episode)
                        print("Saved Model at {}".format(self.model_path + '/model-' + str(episode_count) + '.cptk'))

                    mean_reward = np.mean(self.episode_rewards[-50:])
                    mean_length = np.mean(self.episode_lengths[-50:])
                    mean_value = np.mean(self.episode_mean_values[-50:])

                    self.summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    self.summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    self.summary.value.add(tag='Perf/Value', simple_value=float(mean_value))

                    if FLAGS.train:
                        self.summary.value.add(tag='Losses/Total Loss', simple_value=float(l))
                        self.summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        self.summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
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

                    self.summary_writer.add_summary(self.summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment_global_episode)
                if not FLAGS.train:
                    test_episode_count += 1
                episode_count += 1
