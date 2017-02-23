from multiprocessing import Queue
import tensorflow as tf
from network import GACNetwork
import numpy as np
from predictor import Predictor
from trainer import Trainer
from agent import Agent
from stats import Stats
import time
import flags
FLAGS = tf.app.flags.FLAGS

class Server:
    def __init__(self, nb_actions):

        self.training_q = Queue(maxsize=FLAGS.max_queue_size)
        self.prediction_q = Queue(maxsize=FLAGS.max_queue_size)

        self.network = GACNetwork(nb_actions)
        self.global_step = self.network.global_step

        self.training_step = 0
        self.frame_counter = 0
        self.agents = []
        self.predictors = []
        self.trainers = []
        self.stats = Stats()

    def run(self):
        self.stats.start()

        for i in np.arange(FLAGS.nb_trainers):
            self.trainers.append(Trainer(self, i))
            self.trainers[-1].start()
        for i in np.arange(FLAGS.nb_predictors):
            self.predictors.append(Predictor(self, i))
            self.predictors[-1].start()
        for i in np.arange(FLAGS.nb_concurrent):
            self.agents.append(Agent(self, i))
            self.agents[-1].start()

        while True:
            if self.stats.episode_count.value % FLAGS.checkpoint_interval:
                self.save_model()
            time.sleep(0.01)

    def train(self, updated_episode_buffer, trainer_id):
        self.network.train(updated_episode_buffer, trainer_id)
        self.training_step += 1
        self.frame_counter += updated_episode_buffer.shape[0]
        self.network.increment_global_step()

    def save_model(self):
        self.network.save(self.stats.episode_count.value)
