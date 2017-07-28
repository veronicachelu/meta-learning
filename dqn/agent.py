from threading import Lock

import numpy as np
import tensorflow as tf
from network import DQNetwork
import flags
from collections import deque

FLAGS = tf.app.flags.FLAGS
import random

# Starting threads
main_lock = Lock()


class Agent():
    def __init__(self, game, sess, nb_actions, global_step):
        self.name = "DQN_agent"
        self.model_path = FLAGS.checkpoint_dir
        self.global_episode = global_step
        self.increment_global_episode = self.global_episode.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        self.sess = sess
        self.graph = sess.graph
        self.summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir)
        self.summary = tf.Summary()

        self.q_net = DQNetwork(nb_actions, 'orig')
        self.target_net = DQNetwork(nb_actions, 'target')

        self.targetOps = self.update_target_graph('orig', 'target')
        self.episode_buffer = deque()
        self.actions = np.zeros([nb_actions])
        self.probability_of_random_action = FLAGS.initial_random_action_prob
        self.env = game

    def update_target_graph_tao(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign((1 - FLAGS.TAO) * to_var.value() + FLAGS.TAO * from_var.value()))
        return op_holder

    def update_target_graph(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def train(self):
        minibatch = random.sample(self.episode_buffer, FLAGS.batch_size)
        rollout = np.array(minibatch)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        done = rollout[:, 4]

        target_actionv_values_evaled = self.sess.run(self.target_net.action_values,
                                                     feed_dict={self.target_net.inputs: np.stack(next_observations, axis=0)})
        target_actionv_values_evaled_max = np.max(target_actionv_values_evaled, axis=1)

        target_actionv_values_evaled_new = []

        for i in range(FLAGS.batch_size):
            if done[i]:
                target_actionv_values_evaled_new.append(rewards[i])
            else:
                target_actionv_values_evaled_new.append(
                    rewards[i] + FLAGS.gamma * target_actionv_values_evaled_max[i])

        feed_dict = {self.q_net.target_q: target_actionv_values_evaled_new,
                     self.q_net.inputs: np.stack(observations, axis=0),
                     self.q_net.actions: actions}
        l, _, ms, img_summ = self.sess.run(
            [self.q_net.action_value_loss,
             self.q_net.train_operation,
             self.q_net.merged_summary,
             self.q_net.image_summaries],
            feed_dict=feed_dict)

        self.updateTarget()

        return l / len(rollout), ms, img_summ

    def updateTarget(self):
        for op in self.targetOps:
            self.sess.run(op)

    def play(self, saver):
        episode_count = self.sess.run(self.global_episode)
        total_steps = 0

        print("Starting agent")
        with self.sess.as_default(), self.graph.as_default():
            while total_steps < FLAGS.max_total_steps:
                if episode_count % FLAGS.update_target_estimator_every == 0:
                    self.updateTarget()
                episode_reward = 0
                episode_step_count = 0
                q_values = []
                d = False

                s = self.env.get_initial_state()

                while not d:
                    if random.random() <= self.probability_of_random_action:
                        # choose an action randomly
                        # a = self.env.env.action_space.sample()
                        a = np.random.choice(range(len(self.env.gym_actions)))

                    else:
                        feed_dict = {self.q_net.inputs: [s]}
                        action_values_evaled = self.sess.run(self.q_net.action_values, feed_dict=feed_dict)[0]
                        q_values.append(action_values_evaled)
                        a = np.argmax(action_values_evaled)

                    s1, r, d, info = self.env.step(a)

                    r = np.clip(r, -1, 1)
                    episode_reward += r
                    episode_step_count += 1
                    total_steps += 1
                    self.episode_buffer.append([s, a, r, s1, d])

                    if len(self.episode_buffer) == FLAGS.memory_size:
                        self.episode_buffer.popleft()

                    if total_steps > FLAGS.observation_steps:
                        l, ms, img_summ = self.train()

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                if len(q_values):
                    self.episode_mean_values.append(np.max(np.asarray(q_values)))

                if episode_count % FLAGS.summary_interval == 0 and episode_count != 0 and total_steps > FLAGS.observation_steps:
                    if episode_count % FLAGS.checkpoint_interval == 0:
                        saver.save(self.sess, self.model_path + '/model-' + str(episode_count) + '.cptk',
                                   global_step=self.global_episode)
                        print("Saved Model at {}".format(self.model_path + '/model-' + str(episode_count) + '.cptk'))

                    mean_reward = np.mean(self.episode_rewards[-FLAGS.summary_interval:])
                    mean_length = np.mean(self.episode_lengths[-FLAGS.summary_interval:])
                    mean_value = np.mean(self.episode_mean_values[-FLAGS.summary_interval:])

                    # if episode_count % FLAGS.test_performance_interval == 0:
                    #     won_games = self.episode_rewards[-FLAGS.test_performance_interval:].count(1)
                    #     self.summary.value.add(tag='Perf/Won Games/1000', simple_value=float(won_games))

                    self.summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    self.summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    self.summary.value.add(tag='Perf/Value', simple_value=float(mean_value))

                    self.summary.value.add(tag='Losses/Loss', simple_value=float(l))
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
                    for s in img_summ:
                        self.summary_writer.add_summary(s, episode_count)
                    self.summary_writer.add_summary(self.summary, episode_count)

                    self.summary_writer.flush()

                # gradually reduce the probability of a random actions.
                if self.probability_of_random_action > FLAGS.final_random_action_prob and len(
                        self.episode_buffer) > FLAGS.observation_steps:
                    self.probability_of_random_action -= (
                                                         FLAGS.initial_random_action_prob - FLAGS.final_random_action_prob) / FLAGS.explore_steps
                self.sess.run(self.increment_global_episode)
                episode_count += 1
