import gym
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from datetime import datetime
from agent import Agent

class MountainCar:
    def __init__(self):
        self._output_directory = "videos/{}".format(datetime.now().strftime("%d%m%H%M%S"))
        os.mkdir(self._output_directory)
        self._env = gym.make('MountainCar-v0')
        # gym.logger.set_level(gym.logger.DEBUG)
        self._agent = Agent(self._env)
        self._n_episodes = 1000
        self._current_episode = -1
        self._target_position = 0.5
        self._action_map = ['Left', 'Nothing', 'Right']
    

    def _print_data(self, exploring, current_reward):
        os.system('cls')
        print('Action Taken: {}, {}'.format(action, 'Exploring' if exploring else 'Exploiting'))
        print('Epsilon: {}'.format(self._agent.epsilon))
        print('Current Reward: {}'.format(current_reward))
        print('Total Reward: {}'.format(self._total_reward))

    def run_tick(self):#Is tick the right termonology here?
        self._env.render()
        action, action_was_random = self._agent.get_action(self._state)
        next_state, reward, self._done, _ = self._env.step(action)  
        if next_state[0] >= self._target_position:
            new_reward = 50
        else:
            new_reward = reward * (self._target_position - next_state[0])
        self._total_reward += new_reward
        # print_data(self._action_map[action],action_was_random, new_reward, self.+total_reward, agent.epsilon)
        next_state = self._agent.reshape_input(next_state)
        self._agent.save_to_memory_buffer(self._state, action, reward, next_state, self._done)
        self._state = next_state

        if self._done:
            print("Episode {}/{}, Score: {}".format(self._current_episode, self._n_episodes, self._total_reward))
          # video_recorder.close()

    def close(self):
        self._env.close()

    def train_model(self):
        self._agent.train_model()
        
    def intialise_episode(self, episode_number):
        self._current_episode = episode_number
        self._done = False
        self._inital_state = self._agent.reshape_input(self._env.reset())
        self._total_reward = 0
        self._state = self._inital_state
        # video_recorder = VideoRecorder(env, "{}/test{}.mp4".format(output_directory, episode_number), enabled=False)

    def get_episodes(self):
        return self._n_episodes

    def is_done(self):
        return self._done



solution = MountainCar()

for episode_number in range(solution.get_episodes()):
    solution.intialise_episode(episode_number)
    while not solution.is_done():
       solution.run_tick()
    solution.train_model()

solution.close()