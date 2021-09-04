import numpy as np
from utils import discounted_cumulative_sums


class Buffer:
    def __init__(self, observation_dimensions, buffer_size, gamma=0.99):
        self.observation_buffer = np.zeros(
            (buffer_size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(buffer_size, dtype=np.int32)
        self.reward_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.return_buffer = np.zeros(buffer_size, dtype=np.float32)
        #self.logprobability_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.gamma = gamma
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        #self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)

        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        return (
            self.observation_buffer,
            self.action_buffer,
            self.return_buffer,
            #self.logprobability_buffer,
        )
