import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PolicyGradientAgent:
    def __init__(
        self,
        observation_dimensions,
        action_dimensions,
        buffer,
        policy=None,
        policy_lr=0.001,
        optimizer=keras.optimizers.Adam,
    ):
        self.observation_dimensions = observation_dimensions
        self.action_dimensions = action_dimensions
        self.buffer = buffer
        if policy:
            self.policy = policy
        else:
            self.policy = self.build_policy()
        self.policy_lr = policy_lr
        self.optimizer = optimizer(learning_rate=self.policy_lr)

    def build_policy(self, list_hidden_sizes=[64, 64], activation=tf.tanh):
        observation_input = keras.Input(
            shape=(self.observation_dimensions,), dtype=tf.float32
        )
        x = observation_input
        for size in list_hidden_sizes:
            x = layers.Dense(units=size, activation=activation)(x)
        logits = layers.Dense(units=self.action_dimensions, activation=None)(x)
        policy = keras.Model(inputs=observation_input, outputs=logits)
        return policy

    def logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.action_dimensions) * logprobabilities_all, axis=1
        )
        return logprobability

    @tf.function
    def sample_action(self, observation):
        logits = self.policy(observation)
        probs = tf.nn.softmax(logits)
        log_probs = tf.math.log(probs)
        action = tf.squeeze(tf.random.categorical(log_probs, 1), axis=1)
        return logits, action

    # Train the policy
    @tf.function
    def train_policy(self, observation_buffer, action_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            logits = self.policy(observation_buffer)
            log_probs = self.logprobabilities(logits, action_buffer)
            policy_loss = -tf.reduce_sum(log_probs * return_buffer)
        policy_grads = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(
            zip(policy_grads, self.policy.trainable_variables)
        )
