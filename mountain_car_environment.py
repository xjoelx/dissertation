"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class MountainCarEnv(gym.Env):
    """
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.

    Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).

    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07

    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right

        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.

    Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.

    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.

    Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, reward_function=lambda state, position, velocity : 0 if position >= 0.5 else -1):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.reward_function = reward_function
        self.goal_position = 0.5
        self.goal_velocity = 0
        self.tick_count = 0
        self.episode_length = 100_000

        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        self.tick_count += 1
        new_position, new_velocity = self.state
        new_velocity += (action - 1) * self.force + math.cos(3 * new_position) * (-self.gravity)
        new_velocity = np.clip(new_velocity, -self.max_speed, self.max_speed)
        new_position += new_velocity
        new_position = np.clip(new_position, self.min_position, self.max_position)
        if (new_position == self.min_position and new_velocity < 0):
            new_velocity = 0

        done = (new_position >= self.goal_position and new_velocity >= self.goal_velocity) or self.tick_count > self.episode_length
        reward = self.reward_function(self.state, new_position, new_velocity)

        self.state = (new_position, new_velocity)
        return np.array(self.state), reward, done, {}

    def get_reward_mod(self, state, position, velocity): #Example reward
        reward = 0

        if velocity > self.state[1] >= 0 and velocity >= 0:
            reward = 20

        if velocity < self.state[1] <= 0 and velocity <= 0:
            reward = 20

        if position >= self.goal_position:
            reward += 100_000
        else:
            reward += -25

        return reward

    def get_reward(self, state, position, velocity):
        if position >= self.goal_position:
            return 0
        else:
            return -1

    def get_reward_energy(self, state, position, velocity): #Energy based reward
        return 100 if position >= self.goal_position else 100*((math.sin(3*position) * 0.0025 + 0.5 * velocity * velocity) - (math.sin(3*state[0]) * 0.0025 + 0.5 * state[1] * state[1])) 

    def reset(self):
        # self.state = np.array([-0.7, 0])
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        self.tick_count = 0
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)    

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos-self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))
 
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
