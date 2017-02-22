from threading import Thread
import tensorflow as tf
import numpy as np
import flags
FLAGS = tf.app.flags.FLAGS

class Trainer(Thread):
    def __init__(self, server, thread_id):
        super(Trainer, self).__init__()
        self.setDaemon(True)

        self.id = thread_id
        self.server = server
        self.stop = False

    def run(self):
        while not self.stop:
            batch_size = 0
            while batch_size <= FLAGS.training_min_batch_size:
                states, rewards, actions = self.server.training_q.get()
                if batch_size == 0:
                    batch_states = states
                    batch_rewards = rewards
                    batch_actions = actions
                else:
                    batch_states.extend(states)
                    batch_rewards.extend(rewards)
                    batch_actions.extend(actions)
                batch_size += states.shape[0]

            self.server.train(batch_states, batch_rewards, batch_actions, self.id)

