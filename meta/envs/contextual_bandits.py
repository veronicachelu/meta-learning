import numpy as np

class contextual_bandit():
    def __init__(self):
        self.num_actions = 2
        self.reset()

    def get_state(self):
        self.internal_state = np.random.permutation(self.choices)
        self.state = np.concatenate(np.reshape(self.internal_state, [2, 1, 1, 3]), axis=1)
        return self.state

    def reset(self):
        self.timestep = 0
        color = [np.random.uniform(), np.random.uniform(), np.random.uniform()]
        a = [np.reshape(np.array(color), [1, 1, 3]), np.reshape(1 - np.array(color), [1, 1, 3])]
        self.true = a[0]
        self.choices = a
        return self.get_state()

    def pullArm(self, action):
        self.timestep += 1
        if (self.internal_state[action] == self.true).all() == True:
            reward = 1.0
        else:
            reward = 0.0
        new_state = self.get_state()
        if self.timestep > 99:
            done = True
        else:
            done = False
        return new_state, reward, done, self.timestep


env = contextual_bandit()
# print("The probabilities for the arm are: {}".format(env.bandit))
# print("pulling arm 0")
# for i in range(10):
#     r, d, t = env.pullArm(0)
#     print("The probabilities for the arm are: {}".format(env.bandit))
#     print("reward = {}, terminated = {}, timestep = {}".format(r, d, t))
#
# print("pulling arm 1")
# for i in range(10):
#     r, d, t = env.pullArm(1)
#     print("The probabilities for the arm are: {}".format(env.bandit))
#     print("reward = {}, terminated = {}, timestep = {}".format(r, d, t))
