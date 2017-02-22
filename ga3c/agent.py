from multiprocessing import Process, Queue, Value
import numpy as np
import flags
import gym
import gym_fast_envs
import tensorflow as tf
from atari_environment import AtariEnvironment

FLAGS = tf.app.flags.FLAGS

class Agent(Process):
    def __init__(self, id, prediction_q, training_q):
        super(Agent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q

        gym_env = gym.make(FLAGS.game)
        gym_env.seed(FLAGS.seed)

        self.env = AtariEnvironment(gym_env=gym_env, resized_width=FLAGS.resized_width,
                                    resized_height=FLAGS.resized_height,
                                    agent_history_length=FLAGS.agent_history_length)

        self.nb_actions = len(self.env.gym_actions)
        self.actions = np.arange(self.nb_actions)

        self.wait_q = Queue(maxsize=1)
        self.stop = Value('i', 0)
