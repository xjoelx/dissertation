import cProfile, pstats
import gym
import pyglet
import os
import tensorflow as tf
import numpy as np
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
        # gym.logger.set_level(gym.logger.DEBUG)
        self.NUUMBER_OF_EPISODES = 100_000
        self._agent = Agent(self._env)
        self.RECORDING_FREQ = 50
        self.TARGET_POSITION = 0.5
        self.ACTION_MAP = ['Left', 'Nothing', 'Right']
        self._epoch_writer = CsvWriter("{}/epoch-data.csv".format(self._output_directory), CsvWriter.EPOCH_HEADERS)
        self.AGENT_NAME = "DDQNOriginalRewards"

    def _print_data(self, action, exploring, current_reward):
        os.system('cls')
        print('Action Taken: {}, {}'.format(action, 'Exploring' if exploring else 'Exploiting'))
        print('Current Reward: {}'.format(current_reward))
        print('Total Reward: {}'.format(self._episode_total_reward))

    def run_tick(self):
        action, action_was_random, action_predictions = self._agent.get_action(self._state)
        self._current_episode_actions.append(action)
        next_state, reward, self._done , _ = self._env.step(action)  

        self._episode_total_reward += reward
        self._episode_max_reward = max(self._episode_max_reward, reward)
        self._episode_max_height = max(self._episode_max_height, next_state[0])

        tick_data = [*self._state[0], self.ACTION_MAP[action], action_was_random, reward, *action_predictions]

        self._episode_data.append(tick_data)

        if self._to_record:
            self._env.render()
            self._print_data(self.ACTION_MAP[action],action_was_random, reward)
            self._video_recorder.capture_frame()
        
        next_state = self._agent.reshape_input(next_state)
        self._agent.save_to_memory_buffer(self._state, action, reward, next_state, self._done)
        self._agent.train_model()
     
        self._state = next_state

        if self._done:
            meta_data = self.get_episode_data()
            self._epoch_writer.write_vector(meta_data)
            print("E{}: MR - {}, MH - {}, TR - {}, E - {}".format(*meta_data))

            episode_writer = CsvWriter("{}/e{}.csv".format(self._output_directory, self._current_episode_number), CsvWriter.EPISODE_HEADERS)
            episode_writer.write_matrix(self._episode_data)

            if self._to_record:
                self._video_recorder.close()  

    def get_episode_data(self):
        return [self._current_episode_number,  round(self._episode_max_reward, 5), self._episode_max_height,
                round(self._episode_total_reward,5), round(self._agent.epsilon, 5)]


    def close(self):
        self._env.close()
        self.WEIGHTS_OUTPUT = "{}/{}.hdf5".format(self._output_directory, self.AGENT_NAME)
        self._agent.save(self.WEIGHTS_OUTPUT)

    def train_model(self):
        self._agent.train_model()
        
    def intialise_episode(self, episode_number):
        self._done = False
        self._inital_state = self._agent.reshape_input(self._env.reset())
        self._current_episode_number = episode_number
        self._agent.current_episode = episode_number
        self._episode_max_reward = -1000000
        self._episode_max_height = -5
        self._episode_total_reward = 0
        self._state = self._inital_state
        self._successful = False
        self._current_episode_actions = []
        # self._to_record = self._current_episode_number % self._recording_freq == 0
        self._to_record = False
        self._video_recorder = VideoRecorder(self._env, "{}/episode{}.mp4".format(self._output_directory, self._current_episode_number), enabled=self._to_record)
        self._episode_data = []

    def current_episode_done(self):
        return self._done

    def current_episode_successful(self):
        return self._successful

class MountainCarModyfiedReward(MountainCar):
    def __init__(self):
        super().__init__()

        def reward_function(state, position, velocity):
            reward = 0
            if velocity > state[1] >= 0 and velocity >= 0:
                reward = 20
            if velocity < state[1] <= 0 and velocity <= 0:
                reward = 20
            if position >= 0.5:
                reward += 100_000
            else:
                reward += -25
            return reward

        self._env = MountainCarEnv(reward_function)
        self._env.episode_length = 500
        self.AGENT_NAME = "DDQNModyfiedRewards"

class MountainCarEnergyReward(MountainCar):
    def __init__(self):
        super().__init__()

        def reward_function(state, position, velocity):
            import math
            return 100 if position >= 0.5 else 100*((math.sin(3*position) * 0.0025 + 0.5 * velocity * velocity) - (math.sin(3*state[0]) * 0.0025 + 0.5 * state[1] * state[1])) 


        self._env = MountainCarEnv(reward_function)
        self._env.episode_length = 500
        self.AGENT_NAME = "DDQNModyfiedRewards"



class HandControl(MountainCar):
    def __init__(self):
        super().__init__()
        import keras
        self.model = keras.models.load_model("data/0318135057/weights.hdf5")

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
              
        self._print_data(self._action_map[action],"", reward, state.reshape(1,2))

    def _print_data(self, action, exploring, current_reward, state):
        os.system('cls')
        # print('Action Taken: {}, {}'.format(action, 'Exploring' if exploring else 'Exploiting'))
        # print('Current Reward: {}'.format(current_reward))
        # print('Total Reward: {}'.format(self._episode_total_reward))
        result = self.model.predict(state)
        print(state[0])
        print('Prediction : {}'.format(self._action_map[np.argmax(result)]))
    

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

import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)


ENABLE_PROFILING = False
SHOW_FIRST_EPISODE = False
solution = MountainCar()
start_time = time.time()

try:
    FIRST_EPISODE = int(not SHOW_FIRST_EPISODE)
    while True: 
        for i in range(solution.NUUMBER_OF_EPISODES):
                profiler = cProfile.Profile()
                episode_number = i + FIRST_EPISODE
                solution.intialise_episode(episode_number)

                if ENABLE_PROFILING: profiler.enable()

                while not solution.current_episode_done():
                    solution.run_tick()
                

                if ENABLE_PROFILING:
                    profiler.disable()
                    export_profiling_results(profiler, '{}/e{}-profiling.csv'.format(solution._output_directory, episode_number))
                
                done_time = time.time()
                # print("Episode {} Completed in {}s".format(episode_number, done_time-start_time))
                start_time = done_time
except KeyboardInterrupt:
    solution.close()