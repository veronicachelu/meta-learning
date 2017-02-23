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
                updated_episode_buffer = self.server.training_q.get()
                if batch_size == 0:
                    batch_episode_buffer = updated_episode_buffer
                else:
                    batch_episode_buffer.extend(updated_episode_buffer)
                batch_size += updated_episode_buffer.shape[0]

            self.server.train(updated_episode_buffer, self.id)

