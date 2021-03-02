import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import random
from collections import deque

class Agent:
    def __init__(self, enviroment):
        self.gamma = 0.99
        self.learning_rate = 0.001

        self.epsilon = 0.95
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.training_batch_size = 32

        self.state_space_size = enviroment.observation_space.shape[0]
        self.action_space_size = enviroment.action_space.n

        self.memory_buffer = deque(maxlen=2000)

        self.model = self.__build_model()

    def save_to_memory_buffer(self, state, action, reward, next_state, done):
        self.memory_buffer.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand(0, 1) < self.epsilon:
            return (np.random.randint(self.action_space_size), True)
        else:
            action_values = self.model.predict(np.reshape(state, [1, self.state_space_size]))
            return (np.argmax(action_values), False)

    def train_model(self):
        if len(self.memory_buffer) > self.training_batch_size:
            training_sample =  random.sample(self.memory_buffer, self.training_batch_size)

            states = []
            next_states = []

            for sample in training_sample:
                state, _, _, next_state, _ = sample
                states.append(state)
                next_states.append(next_state)

            states = np.array(states).reshape(self.training_batch_size, 2)
            next_states =  np.array(next_states).reshape(self.training_batch_size, 2)  

            future_rewards = self.model.predict(states)
            next_state_targets = self.model.predict(next_states)

            for i, sample in enumerate(training_sample):
                _, action, reward, _, done = sample
                if done:
                    target_q = reward
                else:
                    target_q = (reward + self.gamma * max(next_state_targets[i]))
                    
                future_rewards[i][action] = target_q

            self.model.fit(states, future_rewards, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def __build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(32, input_dim=self.state_space_size, activation="relu"))
        model.add(keras.layers.Dense(self.action_space_size, activation='linear'))
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
