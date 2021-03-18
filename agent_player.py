import gym
import keras
import numpy as np
from mountain_car_environment import MountainCarEnv

env = MountainCarEnv()
state = env.reset()
model = keras.models.load_model("data/0318135057/weights.hdf5")
done = False
while not done:
    env.render()
    action = np.argmax(model.predict(state.reshape(1,2)))
    state, _, done, _ = env.step(action) # take a random action
env.close()