import gym
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from datetime import datetime
from agent import Agent

def print_data(action, exploring, current_reward, total_reward, epsilon):
    os.system('cls')
    print('Action Taken: {}, {}'.format(action, 'Exploring' if exploring else 'Exploiting'))
    print('Epsilon: {}'.format(epsilon))
    print('Current Reward: {}'.format(current_reward))
    print('Total Reward: {}'.format(total_reward))


output_directory = "videos/{}".format(datetime.now().strftime("%d%m%H%M%S"))
os.mkdir(output_directory)
env = gym.make('MountainCar-v0')
# gym.logger.set_level(gym.logger.DEBUG)
agent = Agent(env)
n_episodes = 1000
target_position = 0.5

action_map = ['Left', 'Nothing', 'Right']

for episode_number in range(n_episodes):
    done = False
    inital_state = agent.reshape_input(env.reset())

    video_recorder = VideoRecorder(env, "{}/test{}.mp4".format(output_directory, episode_number), enabled=False)

    total_reward = 0
    state = inital_state
    while not done:
        # env.render()
        video_recorder.capture_frame()
        action, action_was_random = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)  
        if next_state[0] >= target_position:
            new_reward += 50
        else:
            new_reward = reward * (target_position - next_state[0])
        total_reward += new_reward
        # print_data(action_map[action],action_was_random, new_reward, total_reward, agent.epsilon)
        next_state = agent.reshape_input(next_state)
        agent.save_to_memory_buffer(state, action, reward, next_state, done)
        state = next_state

        if done:
            print("Episode {}/{}, Score: {}".format(episode_number, n_episodes, total_reward))
            video_recorder.close()
        
    agent.train_model()

env.close()