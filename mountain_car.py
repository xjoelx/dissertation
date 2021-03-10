import cProfile, pstats
import gym
import pyglet
import os
import io
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from datetime import datetime
from agent import Agent, ExampleAgent
from mountain_car_environment import MountainCarEnv
import keyboard
from csv_writer import CsvWriter
import time
import cProfile
class MountainCar:
    def __init__(self):
        self._output_directory = "data/{}".format(datetime.now().strftime("%m%d%H%M%S"))
        os.mkdir(self._output_directory)
        self._env = MountainCarEnv()
        # self._env = gym.make('MountainCar-v0')
        gym.logger.set_level(gym.logger.DEBUG)
        self.number_of_episodes = 1000
        self._agent = Agent(self._env)
        self._recording_freq = 100
        self._current_episode_number = 0
        self._target_position = 0.5
        self._episode_max_reward = -1000000
        self._episode_total_reward = 0
        self._action_map = ['Left', 'Nothing', 'Right']
        self._writer = CsvWriter("{}/data.csv".format(self._output_directory), self.get_csv_headers())
        

    def _print_data(self, action, exploring, current_reward):
        os.system('cls')
        print('Action Taken: {}, {}'.format(action, 'Exploring' if exploring else 'Exploiting'))
        print('Current Reward: {}'.format(current_reward))
        print('Total Reward: {}'.format(self._episode_total_reward))

    def run_tick(self):
    
        if self._to_record:
            self._env.render()
            self._video_recorder.capture_frame()
        action, action_was_random = self._agent.get_action(self._state)
        self._current_episode_actions.append(action)
        next_state, reward, self._done , _ = self._env.step(action)  

        self._episode_total_reward += reward
        self._episode_max_reward = max(self._episode_max_reward, reward)
        # self._print_data(self._action_map[action],action_was_random, reward)
        next_state = self._agent.reshape_input(next_state)
        self._agent.save_to_memory_buffer(self._state, action, reward, next_state, self._done)
        self._state = next_state

        if self._done:
            self._writer.write_data(self.get_data_to_write())
            if self._to_record:
                self._video_recorder.close()  

    def get_csv_headers(self):
        return ["Episode Number", "Maximum Reward", "Total Reward", "Exploring Rate"]

    def get_data_to_write(self):
        return [self._current_episode_number,  self._episode_max_reward,
                self._episode_total_reward, self._agent.epsilon]

    def close(self):
        self._env.close()

    def train_model(self):
        self._agent.train_model()
        
    def intialise_episode(self, episode_number):
        self._current_episode_number = episode_number
        self._done = False
        self._inital_state = self._agent.reshape_input(self._env.reset())
        self._agent.current_episode = episode_number
        self._episode_max_reward = -1000000
        self._episode_total_reward = 0
        self._state = self._inital_state
        self._successful = False
        self._current_episode_actions = []
        self._to_record = self._current_episode_number % self._recording_freq == 0
        self._video_recorder = VideoRecorder(self._env, "{}/episode{}.mp4".format(self._output_directory, self._current_episode_number), enabled=self._to_record)


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
        state, reward, self._done, _ = self._env.step(action)
        if state[0] >= self._target_position:
            self._successful = True
              
        self._print_data(self._action_map[action],"", reward)
    

# https://qxf2.com/blog/saving-cprofile-stats-to-a-csv-file/
def export_profiling_results(profiler, file_name):
    result = io.StringIO()
    pstats.Stats(profiler,stream=result).sort_stats("cumtime").print_stats()
    result=result.getvalue()
    # chop the string into a csv-like buffer
    result='ncalls'+result.split('ncalls')[-1]
    result='\n'.join([','.join(line.rstrip().split(None,5)) for line in result.split('\n')])
    # save it to disk
    
    with open(file_name, 'w+') as f:
        #f=open(result.rsplit('.')[0]+'.csv','w')
        f.write(result)
        f.close()


solution = MountainCar()
start_time = time.time()
done = False

while not done:
    try:
        profiler = cProfile.Profile()
        episode_number = i + 1
        solution.intialise_episode(episode_number)
        profiler.enable()
        while not solution.current_episode_done():
            solution.run_tick()
        solution.train_model()
        profiler.disable()
        
        export_profiling_results(profiler, '{}/episode{}.csv'.format(solution._output_directory, episode_number))
        
        done_time = time.time()
        print("Episode {} Completed in {}s".format(episode_number, done_time-start_time))
        start_time = done_time
    except KeyboardInterrupt:
        done = True
        solution.close()