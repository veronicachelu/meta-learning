from threading import Thread
import tensorflow as tf
import numpy as np
import flags
import time
FLAGS = tf.app.flags.FLAGS

class Predictor(Thread):
    def __init__(self, server, thread_id):
        super(Predictor, self).__init__(name="Predictor_{}".format(thread_id))
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
                if self.server.prediction_q.empty():
                    break
                print("Predictor_{} gets a new prediction form the prediction queue".format(self.id))
                agents_ids[i], states[i] = self.server.prediction_q.get()

            if i > 0:
                pi, v = self.server.network.predict(states[:i])

                for j in np.arange(i):
                    print("Predictor_{} puts a new prediction in the agent {} wait queue".format(self.id, agents_ids[j]))
                    self.server.agents[agents_ids[j]].wait_q.put((pi[j], v[j]))
            time.sleep(0.01)
