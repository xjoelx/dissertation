import gym
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from datetime import datetime
from agent import Agent
import keyboard
from csv_writer import CsvWriter
class MountainCar:
    def __init__(self):
        self._output_directory = "data/{}".format(datetime.now().strftime("%d%m%H%M%S"))
        os.mkdir(self._output_directory)
        self._env = gym.make('MountainCar-v0')
        # gym.logger.set_level(gym.logger.DEBUG)
        self._agent = Agent(self._env)
        self._n_episodes = 1000
        self._current_episode_number = -1
        self._target_position = 0.5
        self._episode_max_reward = -1000000
        self._episode_total_reward = 0
        self._action_map = ['Left', 'Nothing', 'Right']
        self._writer = CsvWriter("{}/data.csv".format(self._output_directory), self.get_csv_headers())
        

    def _print_data(self, action, exploring, current_reward):
        os.system('cls')
        print('Action Taken: {}, {}'.format(action, 'Exploring' if exploring else 'Exploiting'))
        print('Epsilon: {}'.format(self._agent.epsilon))
        print('Current Reward: {}'.format(current_reward))
        print('Total Reward: {}'.format(self._episode_total_reward))

    def run_tick(self):#Is tick the right termonology here?
        self._env.render()
        action, action_was_random = self._agent.get_action(self._state)
        self._current_episode_actions.append(action)
        next_state, reward, self._done, _ = self._env.step(action)  
        if next_state[0] >= self._target_position:
            new_reward = 1000
            self._successful = True
        else:
            new_reward = reward * (self._target_position - next_state[0])
        self._episode_total_reward += new_reward
        self._episode_max_reward = new_reward if new_reward > self._episode_max_reward else self._episode_max_reward
        # print_data(self._action_map[action],action_was_random, new_reward, self.+total_reward, agent.epsilon)
        next_state = self._agent.reshape_input(next_state)
        self._agent.save_to_memory_buffer(self._state, action, reward, next_state, self._done)
        self._state = next_state

        if self._done:
            self._writer.write_data(self.get_data_to_write())

    def get_csv_headers(self):
        return ["Episode Number", "Maximum Reward", "Total Reward", "Exploring Rate"]

    def get_data_to_write(self):
        return [self._current_episode_number,  self._episode_max_reward,
                self._episode_total_reward, self._agent.epsilon]

    def replay_episode(self):
        self._env.reset()
        video_recorder = VideoRecorder(self._env, "{}/episode{}.mp4".format(self._output_directory, episode_number), enabled=True)
        for action_taken in self._current_episode_actions:
            self._env.render()
            video_recorder.capture_frame()
            self._env.step(action_taken)
        video_recorder.close()

    def close(self):
        self._env.close()

    def train_model(self):
        self._agent.train_model()
        
    def intialise_episode(self, episode_number):
        self._current_episode_number = episode_number
        self._done = False
        self._inital_state = self._agent.reshape_input(self._env.reset())
        self._episode_max_reward = -1000000
        self._episode_total_reward = 0
        self._state = self._inital_state
        self._successful = False
        self._current_episode_actions = []

    def get_episode_count(self):
        return self._n_episodes

    def current_episode_done(self):
        return self._done

    def current_episode_successful(self):
        return self._successful

class HandControl(MountainCar):
    def run_tick(self):
        self._env.render()
        if keyboard.is_pressed("a"):
            action = 0
        elif keyboard.is_pressed("d"):
            action = 2
        else:
            action = 1
        self._current_episode_actions.append(action)
        state, _, self._done, _ = self._env.step(action)
        if state[0] >= self._target_position:
            self._successful = True


solution = MountainCar()
# solution = HandControl()

for i in range(solution.get_episode_count()):
    episode_number = i + 1
    solution.intialise_episode(episode_number)
    while not solution.current_episode_done():
       solution.run_tick()

    solution.train_model()
    
    if solution.current_episode_successful():
        solution.replay_episode()

solution.close()