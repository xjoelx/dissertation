import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import random
from collections import deque

class Agent:
    def __init__(self, environment):
        self.gamma = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.exploration_ticks = 10000
        self.training_batch_size = 32

        self.state_space_size = environment.observation_space.shape[0]
        self.action_space_size = environment.action_space.n

        self.memory_buffer = deque(maxlen=50000)

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
        if len(self.memory_buffer) > self.exploration_ticks:
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

            self.model.fit(states, future_rewards, batch_size=self.training_batch_size, verbose=0)

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

class ExampleAgent(Agent):
    def __init__(self, environment, batch_size=32, memory_size=50000):
        self.state_space_size = environment.observation_space.shape[0]
        self.action_space_size = environment.action_space.n
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.training = 10000  # training after 10000 env steps
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_space_size, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def save_to_memory_buffer(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return (random.randrange(self.action_space_size), True)
        else:
            return (np.argmax(self.model.predict(state)[0]), False)

    def train_model(self):
        # Updates the online network weights after enough data is collected
        if self.training >= len(self.memory):
            return

        # Samples a batch from the memory
        random_batch = random.sample(self.memory, self.batch_size)

        state = np.zeros((self.batch_size, self.state_space_size))
        next_state = np.zeros((self.batch_size, self.state_space_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = random_batch[i][0]
            action.append(random_batch[i][1])
            reward.append(random_batch[i][2])
            next_state[i] = random_batch[i][3]
            done.append(random_batch[i][4])

        # Batch prediction to save compute costs
        target = self.model.predict(state)
        target_next = self.model(next_state)

        for i in range(len(random_batch)):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(
            np.array(state),
            np.array(target),
            batch_size=self.batch_size,
            verbose=0
        )

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
