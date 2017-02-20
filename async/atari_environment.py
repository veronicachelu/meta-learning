from collections import deque

import numpy as np
from skimage.color import rgb2gray
# from skimage.transform import resize
from PIL import Image

class AtariEnvironment(object):
    def __init__(self, gym_env, resized_width, resized_height, agent_history_length):
        self.env = gym_env
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.agent_history_length = agent_history_length

        self.gym_actions = range(gym_env.action_space.n)
        if (gym_env.spec.id == "Pong-v0" or gym_env.spec.id == "Breakout-v0"):
            print("Doing workaround for pong or breakout")
            # Gym returns 6 possible actions for breakout and pong.
            # Only three are used, the rest are no-ops. This just lets us
            # pick from a simplified "LEFT", "RIGHT", "NOOP" action space.
            self.gym_actions = [1, 2, 3]

        # Screen buffer of size AGENT_HISTORY_LENGTH to be able
        # to build state arrays of size [1, AGENT_HISTORY_LENGTH, width, height]
        self.state_buffer = deque()

    def get_initial_state(self):
        # Clear the state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        for i in range(self.agent_history_length - 1):
            self.state_buffer.append(x_t)
        return s_t

    def get_preprocessed_frame(self, observation):
        # gray = 0.2125 * observation[..., 0]
        # gray[:] += 0.7154 * observation[..., 1]
        # gray[:] += 0.0721 * observation[..., 2]

        #
        #     return gray
        lum = Image.fromarray(observation)
        lum = lum.convert('L')

        lum = lum.resize((self.resized_width, self.resized_height))
        pix = np.array(lum).astype(float) / 255
        # return self.color2gray(observation).resize((self.resized_width, self.resized_height))
        return pix

    def step(self, action_index):
        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1 = self.get_preprocessed_frame(x_t1)

        previous_frames = np.array(self.state_buffer)
        previous_frames = np.transpose(previous_frames, axes=(1, 2, 0))
        s_t1 = np.empty((self.resized_height, self.resized_width, self.agent_history_length))
        s_t1[:, :, :self.agent_history_length - 1] = previous_frames
        s_t1[:, :, self.agent_history_length - 1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info

    # def color2gray(self, rgb):
    #     if rgb.ndim == 2:
    #         return np.ascontiguousarray(rgb)
    #
    #     rgb = self.prepare_colorarray(rgb[..., :3])
    #
    #     gray = 0.2125 * rgb[..., 0]
    #     gray[:] += 0.7154 * rgb[..., 1]
    #     gray[:] += 0.0721 * rgb[..., 2]
    #
    #     return gray
    #
    # def prepare_colorarray(self, arr):
    #     """Check the shape of the array and convert it to
    #     floating point representation.
    #
    #     """
    #     arr = np.asanyarray(arr)
    #
    #     if arr.ndim not in [3, 4] or arr.shape[-1] != 3:
    #         msg = ("the input array must be have a shape == (.., ..,[ ..,] 3)), " +
    #                "got (" + (", ".join(map(str, arr.shape))) + ")")
    #         raise ValueError(msg)
    #
    #     return arr.astype(float)