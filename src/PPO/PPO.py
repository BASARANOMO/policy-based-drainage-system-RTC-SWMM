# PPO Clipped

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations

class ClippedPPOAgent:
    def __init__(
        self,
        observation_dimensions,
        action_dimensions,
        buffer,
        actor=None,
        critic=None,
        policy_lr=0.001,
        optimizer=keras.optimizers.Adam,
    ):
        self.observation_dimensions = observation_dimensions
        self.action_dimensions = action_dimensions
        self.buffer = buffer
        if actor:
            self.actor = actor
        else:
            self.actor = self.build_default_policy()
        if critic:
            self.critic = critic
        else:
            self.critic = self.build_default_policy()
        self.policy_lr = policy_lr
        self.optimizer = optimizer(learning_rate=self.policy_lr)
