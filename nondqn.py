import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from mountain_car_environment import MountainCarEnv

env = MountainCarEnv()
actions = list(range(env.action_space.n))
position_states = 12
velocity_states = 12
epsilon = 1
ALPHA = 0.5
GAMMA = 0.99
EPSILON_MIN = 0
EPSILON_DECAY = 0.95

position_space = np.linspace(-1.2, 0.6, position_states)
velocity_space = np.linspace(-0.07, 0.07, velocity_states)

def get_state(observation):
    position, velocity = observation
    position_bin = int(np.digitize(position, position_space))
    velocity_bin = int(np.digitize(velocity, velocity_space))
    return (position_bin, velocity_bin)

def get_action(state, q_function):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        action_values = np.array([q_function[state[0], state[1], a] for a in actions])
        return np.argmax(action_values)

def save_data(data):
    arr_reshaped = data.reshape(data.shape[0], -1)
    np.savetxt("q.txt", arr_reshaped)

def load_data():
    loaded_arr = np.loadtxt("q.txt")
    return loaded_arr.reshape( loaded_arr.shape[0], loaded_arr.shape[1] // env.action_space.n, env.action_space.n)

q = np.zeros((position_states + 1, velocity_states + 1, env.action_space.n))
total_rewards = []

number_of_episodes = 250
for i in range(number_of_episodes):
    state = get_state(env.reset())
    total_reward = 0
    for _ in range(env.episode_length):
        action = get_action(state, q)
        raw_next_state, reward, done, _ = env.step(action)
        next_state = get_state(raw_next_state)
        total_reward += reward
        td_target = reward + GAMMA * max([q[next_state[0], next_state[1], a] for a in actions])
        q[state[0], state[1], action] = q[state[0], state[1], action]  + ALPHA * (td_target - q[state[0], state[1], action])
        state = next_state
    epsilon = epsilon * EPSILON_DECAY
    total_rewards.append(total_reward)
    print("Epsiode {} completed with TR {}".format(i, total_reward))
env.close()
save_data(q)

plt.plot(range(len(total_rewards)), total_rewards, label="Total Reward per episode")
plt.gca().invert_yaxis()
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()