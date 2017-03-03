import random

import gym
import gym_fast_envs

# xavier_init = tf.contrib.layers.xavier_initializer()

env = gym.make('Gridworld-v0')
env.reset()
for _ in range(90):
    env.step(random.randint(0, env.action_space.n - 1))
    env.render()
