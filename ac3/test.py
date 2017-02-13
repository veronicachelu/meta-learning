import gym
import gym_ple
import flags
import threading
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

class t(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name)
        self.thread_id, self.env = args
    def run(self):
        self.env.reset()
        print("dadsa")


envs = [gym.make(FLAGS.EXPERIMENT) for _ in range(2)]

actor_learner_threads = [t(args=(thread_id, envs[thread_id])) for thread_id in
                             range(2)]
for t in actor_learner_threads:
    t.start()


while True:
    for env in envs:
        env.render()


for t in actor_learner_threads:
    t.join()
