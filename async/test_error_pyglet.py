import random

import gym

# xavier_init = tf.contrib.layers.xavier_initializer()

env = gym.make('Pong-v0')
env.reset()
for _ in range(90):
    env.step(random.randint(0, env.action_space.n - 1))
    env.render()
