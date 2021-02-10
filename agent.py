import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from collections import deque

class Agent:
    def __init__(self, enviroment):
        self.gamma = 0.99
        self.learning_rate = 0.001

        self.epsilon = 0.95
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01
        self.training_batch_size = 32

        self.state_space_size = enviroment.observation_space.shape[0]
        self.action_space_size = enviroment.action_space.n

        self.memory_buffer = deque(maxlen=2000)

        self.model = self.__build_model()

    def save_to_memory_buffer(self, state, action, reward, next_state, done):
        self.memory_buffer.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return (random.randrange(self.action_space_size), True)
        else:
            action_values = self.model.predict(np.reshape(state, [1, self.state_space_size]))
            return (np.argmax(action_values), False)

    def train_model(self):
        if len(self.memory_buffer) > self.training_batch_size:
            training_sample = random.sample(self.memory_buffer, self.training_batch_size)

            for state, action, reward, next_state, done in training_sample:
                if done:
                    target_q = reward
                else:
                    target_q = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
                
                future_rewards = self.model.predict(state)
                future_rewards[0][action] = target_q

                self.model.fit(state, future_rewards, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def __build_model(self):
        model = keras.models.Sequential()
        model.add(layers.Dense(32, input_dim=self.state_space_size, activation="relu"))
        model.add(layers.Dense(self.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def get_state_count(self):
        return self.state_space_size

    def get_action_count(self):
        return self.action_space_size

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def reshape_input(self, input):
        return np.reshape(input, [1, self.state_space_size])
