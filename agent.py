import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import random
from collections import deque

class Agent:
    def __init__(self, environment):
        self.UPDATE_TRAGET_EVERY = 10
        self.GAMMA = 1
        self.LEARNING_RATE = 0.001
        self.EPSILON_DECAY = 0.9995
        self.EPSILON_MIN = 0.01
        self.EXPLORATION_EPISODES = 50
        self.TRAINING_BATCH_SIZE = 32
        self.STATE_SPACE_SIZE = environment.observation_space.shape[0]
        self.ACTION_SPACE_SIZE = environment.action_space.n
        self.EPISODE_LENGTH = environment.episode_length
        self.current_episode = 0
        self.current_update_count = 0
        self.epsilon = 1

        self.memory_buffer = deque(maxlen=50000)

        self.model = self.__build_model()
        self.target_model = self.__build_model()
        self.target_model.set_weights(self.model.get_weights())

    def save_to_memory_buffer(self, state, action, reward, next_state, done):
        self.memory_buffer.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return (random.randrange(self.ACTION_SPACE_SIZE), True, [-1, -2, -3])
        else:
            action_values = self.model(np.reshape(state, [1, self.STATE_SPACE_SIZE]))
            return (np.argmax(action_values), False, action_values)

    def train_model(self):
        if self.current_episode >= self.EXPLORATION_EPISODES:
            training_sample =  random.sample(self.memory_buffer, self.TRAINING_BATCH_SIZE)

            states = []
            next_states = []

            for sample in training_sample:
                state, _, _, next_state, _ = sample
                states.append(state)
                next_states.append(next_state)

            states = np.array(states).reshape(self.TRAINING_BATCH_SIZE, 2)
            next_states =  np.array(next_states).reshape(self.TRAINING_BATCH_SIZE, 2)  

            future_rewards = np.array(self.model(states))
            next_state_targets = np.array(self.target_model(next_states))

            for i, sample in enumerate(training_sample):
                _, action, reward, _, done = sample
                if done:
                    target_q = reward
                else:
                    target_q = (reward + self.GAMMA * max(next_state_targets[i]))
                    
                future_rewards[i][action] = target_q

            self.model.fit(states, future_rewards, batch_size=self.TRAINING_BATCH_SIZE, verbose=0, epochs=50)

            self.epsilon = max(self.epsilon * self.EPSILON_DECAY, self.EPSILON_MIN)
            self.current_update_count += 1
            if self.current_update_count >= self.UPDATE_TRAGET_EVERY:
                self.current_update_count = 0
                self.target_model.set_weights(self.model.get_weights())

    def __build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.STATE_SPACE_SIZE, activation="relu"))
        model.add(keras.layers.Dense(24, activation="relu"))
        model.add(keras.layers.Dense(self.ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.LEARNING_RATE))
        return model

    def _get_callback(self):
        return tf.keras.callbacks.ModelCheckpoint(filepath=self.WEIGHTS_OUTPUT,
                                                    save_weights_only=True,
                                                    verbose=1)

    def get_state_count(self):
        return self.STATE_SPACE_SIZE

    def get_action_count(self):
        return self.ACTION_SPACE_SIZE

    def load(self, name):
        self.model.load_weights(name)

    def save(self, output_directory):
        self.model.save(output_directory)

    def reshape_input(self, input):
        return np.reshape(input, [1, self.STATE_SPACE_SIZE])

class ExampleAgent(Agent):
    def __init__(self, environment):
        self.STATE_SPACE_SIZE = environment.observation_space.shape[0]
        self.ACTION_SPACE_SIZE = environment.action_space.n
        self.TRAINING_BATCH_SIZE = 1000
        self.EXPLORATION_TICKS = 50 * environment.episode_length
        self.GAMMA = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.EPSILON_MIN = 0.01
        self.EPSILON_DECAY = 0.995
        self.LEARNING_RATE = 0.001
        
        self.model = self._build_model()
        self.memory_buffer = deque(maxlen=50_000)

    def _build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.STATE_SPACE_SIZE, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.LEARNING_RATE))
        return model

    def save_to_memory_buffer(self, state, action, reward, next_state, done):
        self.memory_buffer.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return (random.randrange(self.ACTION_SPACE_SIZE), True, [-99, -99, -99])
        else:
            predictions = self.model.predict(state)[0]
            return (np.argmax(predictions), False, predictions)

    def train_model(self):
        # Updates the online network weights after enough data is collected
        if self.EXPLORATION_TICKS >= len(self.memory_buffer):
            return

        # Samples a batch from the memory
        random_batch = random.sample(self.memory_buffer, self.TRAINING_BATCH_SIZE)

        state = np.zeros((self.TRAINING_BATCH_SIZE, self.STATE_SPACE_SIZE))
        next_state = np.zeros((self.TRAINING_BATCH_SIZE, self.STATE_SPACE_SIZE))
        action, reward, done = [], [], []

        for i in range(self.TRAINING_BATCH_SIZE):
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
                target[i][action[i]] = reward[i] + self.GAMMA * (np.amax(target_next[i]))

        self.model.fit(
            np.array(state),
            np.array(target),
            batch_size=self.TRAINING_BATCH_SIZE,
            verbose=0
        )

        self.epsilon = max(self.EPSILON_MIN, self.epsilon * self.EPSILON_DECAY)
