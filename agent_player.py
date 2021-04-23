import gym
import keras
import os
import numpy as np
from mountain_car_environment import MountainCarEnv

data_directory = "0401085335"
file_name = ""

for file in os.listdir("data/{}".format(data_directory)):
    if file.endswith(".hdf5"):
        file_name = file

env = MountainCarEnv()
state = env.reset()
model = keras.models.load_model("data/{}/{}".format(data_directory, file_name)) #Trained with example reward
# model = keras.models.load_model("data/0326070330/weights.hdf5") #Trained with energy reward
done = False
while not done:
    env.render()
    action = np.argmax(model.predict(state.reshape(1,2)))
    state, _, done, _ = env.step(action) # take a random action
env.close()