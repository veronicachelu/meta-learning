from threading import Thread
import tensorflow as tf
import numpy as np
import flags
FLAGS = tf.app.flags.FLAGS

class Predictor(Thread):
    def __init__(self, server, thread_id):
        super(Predictor, self).__init__()
        self.setDaemon(True)

        self.id = thread_id
        self.server = server
        self.stop = False

    def run(self):
        agents_ids = np.zeros(FLAGS.prediction_batch_size, dtype=np.uint16)
        states = np.zeros(
            (FLAGS.prediction_batch_size, FLAGS.resized_height, FLAGS.resized_width, FLAGS.agent_history_length),
            dtype=np.float32)

        while not self.stop:
            for i in np.arange(FLAGS.prediction_batch_size):
                if not self.server.prediction_q.empty():
                    agents_ids[i], states[i] = self.server.prediction_q.get()

            pi, v = self.server.network.predict(states)

            for i in np.arange(len(states)):
                self.server.agents[agents_ids[i]].wait_q.put((pi[i], v[i]))

